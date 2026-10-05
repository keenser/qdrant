use std::collections::{HashMap, HashSet};
use std::time::Duration;

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use storage::rbac::{
    Access, CollectionAccess, CollectionAccessList, CollectionAccessMode, GlobalAccessMode,
};
use tokio::sync::Mutex;

use crate::settings::IamConfig;

/// Hard cap on the number of cached introspection results.
///
/// The key is the token itself, so without a bound the map would grow with every
/// distinct token the process ever sees. On insert we first drop expired entries
/// and, if that is not enough, evict the entry closest to expiry.
const CACHE_MAX_ENTRIES: usize = 10_000;

/// A cached introspection result.
///
/// The entry stores the point in time it must be considered stale, which is the
/// earlier of the configured cache TTL and the token's own `exp`. This keeps a
/// token from being served from the cache after Hydra would consider it expired.
struct CachedIntrospection {
    result: IntrospectionResult,
    expires_at: DateTime<Utc>,
}

/// A single opaque token introspection result, as returned by the Ory Hydra
/// `/oauth2/introspect` endpoint.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct IntrospectionResult {
    /// Whether the token is currently active.
    pub active: bool,

    /// The client identifier for the OAuth2 client that requested the token.
    #[serde(default)]
    pub client_id: Option<String>,

    /// The subject of the token.
    #[serde(default)]
    pub sub: Option<String>,

    /// A space-delimited list of scopes associated with the token.
    #[serde(default)]
    pub scope: Option<String>,

    /// Token type, e.g. `access_token`.
    #[serde(default)]
    pub token_type: Option<String>,

    /// Expiration time (seconds since UNIX epoch).
    #[serde(default)]
    pub exp: Option<u64>,

    /// Issued at (seconds since UNIX epoch).
    #[serde(default)]
    pub iat: Option<u64>,

    /// Not before (seconds since UNIX epoch).
    #[serde(default)]
    pub nbf: Option<u64>,

    /// Issuer identifier.
    #[serde(default)]
    pub iss: Option<String>,

    /// Intended audience(s) of the token.
    #[serde(default)]
    pub aud: Option<Vec<String>>,

    /// The username of the resource owner.
    #[serde(default)]
    pub username: Option<String>,
}

impl IntrospectionResult {
    /// The parsed set of scopes.
    pub fn scopes(&self) -> HashSet<String> {
        self.scope
            .as_deref()
            .map(|s| s.split_whitespace().map(str::to_owned).collect())
            .unwrap_or_default()
    }
}

/// Identity metadata extracted from an introspection result, used for audit
/// logging. Carries the OAuth2 `client_id` (the service/app that requested the
/// token) and, when present, the `sub` (resource owner).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IamIdentity {
    /// The OAuth2 client that requested the token, if the introspection
    /// response provided a non-empty one.
    pub client_id: Option<String>,
    /// The resource owner subject, if provided by the token.
    pub sub: Option<String>,
}

/// A client that introspects opaque bearer tokens against Spirit IAM
/// (Ory Hydra `/oauth2/introspect`).
#[derive(Clone)]
pub struct IamClient {
    inner: std::sync::Arc<IamClientInner>,
}

struct IamClientInner {
    introspection_url: String,
    client_id: String,
    client_secret: String,
    cache_ttl: ChronoDuration,
    audience: HashSet<String>,
    http: reqwest::Client,
    cache: Mutex<HashMap<String, CachedIntrospection>>,
}

/// Error returned by the IAM client.
#[derive(Debug, thiserror::Error)]
pub enum IamError {
    #[error("IAM introspection request failed: {0}")]
    Request(String),

    #[error("IAM introspection returned an unexpected status: {0}")]
    UnexpectedStatus(reqwest::StatusCode),

    #[error("Failed to parse IAM introspection response: {0}")]
    MalformedResponse(String),

    #[error("Token is not active")]
    Inactive,

    #[error("Token audience is not trusted")]
    UntrustedAudience,

    #[error("Token grants no recognized Qdrant scope")]
    NoScope,
}

impl IamClient {
    /// Create a new IAM client from the given configuration.
    pub fn new(config: &IamConfig) -> Self {
        let introspection_url = format!("{}/oauth2/introspect", config.url.trim_end_matches('/'));
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_sec))
            .build()
            .expect("Failed to build IAM HTTP client");

        // `config` is validated (`cache_ttl_sec >= 1`) before an `IamClient` is
        // constructed from it in `AuthKeys::try_create`; tests that bypass that
        // path are expected to pass a valid `cache_ttl_sec` as well.
        let cache_ttl = ChronoDuration::try_seconds(config.cache_ttl_sec as i64)
            .expect("cache_ttl_sec is validated to fit in i64 seconds");

        Self {
            inner: std::sync::Arc::new(IamClientInner {
                introspection_url,
                client_id: config.client_id.clone(),
                client_secret: config.client_secret.clone(),
                cache_ttl,
                audience: config.audience.iter().cloned().collect(),
                http,
                cache: Mutex::new(Default::default()),
            }),
        }
    }

    /// Introspect an opaque token, using the in-memory cache when possible.
    pub async fn introspect(&self, token: &str) -> Result<IntrospectionResult, IamError> {
        let now = Utc::now();

        // Fast path: serve from cache when a non-expired entry exists. The
        // entry's `expires_at` already accounts for both the configured cache
        // TTL and the token's own `exp`, so a token can never be served from
        // the cache past the point Hydra itself would consider it expired.
        {
            let cache = self.inner.cache.lock().await;
            if let Some(entry) = cache.get(token)
                && entry.expires_at >= now
            {
                return Ok(entry.result.clone());
            }
        }

        let result = self.introspect_remote(token).await?;

        // Cache only positive results; negative results are cheap to recompute
        // and may flip back to active at any moment.
        if result.active {
            // The cache entry must not outlive the token itself.
            let expires_at = match result.exp {
                Some(exp) => {
                    let token_exp = DateTime::<Utc>::from_timestamp(exp as i64, 0).unwrap_or(now);
                    (now + self.inner.cache_ttl).min(token_exp)
                }
                None => now + self.inner.cache_ttl,
            };

            // Already expired (or expiring right now) by the token's own `exp`:
            // not worth caching at all.
            if expires_at > now {
                let mut cache = self.inner.cache.lock().await;
                evict_stale_or_oldest(&mut cache, now);
                cache.insert(
                    token.to_string(),
                    CachedIntrospection {
                        result: result.clone(),
                        expires_at,
                    },
                );
            }
        }

        Ok(result)
    }

    /// Introspect an opaque token and map the granted scopes to Qdrant
    /// [`Access`] rights, together with the identity metadata used for audit
    /// logging.
    ///
    /// Returns an error if the token is not active, its audience is not trusted,
    /// or it grants no recognized Qdrant scope.
    pub async fn validate(&self, token: &str) -> Result<(Access, IamIdentity), IamError> {
        let result = self.introspect(token).await?;

        if !result.active {
            return Err(IamError::Inactive);
        }

        if !self.inner.audience.is_empty() {
            let granted = result.aud.as_deref().unwrap_or_default();
            let trusted = granted.iter().any(|aud| self.inner.audience.contains(aud));
            if !trusted {
                return Err(IamError::UntrustedAudience);
            }
        }

        let access = access_from_scopes(&result.scopes()).ok_or(IamError::NoScope)?;
        let identity = IamIdentity {
            // An empty `client_id` is not useful for audit attribution either;
            // normalize it to `None` like every other optional identity field.
            client_id: result.client_id.filter(|s| !s.is_empty()),
            sub: result.sub.clone(),
        };

        Ok((access, identity))
    }

    async fn introspect_remote(&self, token: &str) -> Result<IntrospectionResult, IamError> {
        let body =
            serde_urlencoded::to_string([("token", token), ("token_type_hint", "access_token")])
                .map_err(|e| IamError::Request(e.to_string()))?;

        let response = self
            .inner
            .http
            .post(&self.inner.introspection_url)
            .basic_auth(&self.inner.client_id, Some(&self.inner.client_secret))
            .header(
                reqwest::header::CONTENT_TYPE,
                "application/x-www-form-urlencoded",
            )
            .body(body)
            .send()
            .await
            .map_err(|e| IamError::Request(e.to_string()))?;

        if !response.status().is_success() {
            return Err(IamError::UnexpectedStatus(response.status()));
        }

        response
            .json::<IntrospectionResult>()
            .await
            .map_err(|e| IamError::MalformedResponse(e.to_string()))
    }
}

/// Keep the introspection cache bounded.
///
/// First drops every entry that has already expired. If the cache is still at
/// capacity, evicts the single entry closest to expiry to make room — cheaper
/// than a full LRU and good enough for a cache whose entries expire on their
/// own within minutes.
fn evict_stale_or_oldest(cache: &mut HashMap<String, CachedIntrospection>, now: DateTime<Utc>) {
    cache.retain(|_, entry| entry.expires_at > now);

    if cache.len() >= CACHE_MAX_ENTRIES
        && let Some(oldest_key) = cache
            .iter()
            .min_by_key(|(_, entry)| entry.expires_at)
            .map(|(key, _)| key.clone())
    {
        cache.remove(&oldest_key);
    }
}

/// Build a Qdrant [`Access`] from a set of IAM scopes.
///
/// Supported scope conventions:
/// - `qdrant:manage`                  -> global manage (full) access
/// - `qdrant:read`                    -> global read-only access
/// - `qdrant:r`                       -> global read-only access (alias of `qdrant:read`)
/// - `qdrant:<collection>:rw`         -> read/write access to a specific collection
/// - `qdrant:<collection>:r`          -> read-only access to a specific collection
///
/// There is no global "read/write but not manage" scope: Qdrant's global access
/// is a binary [`GlobalAccessMode::Read`]/[`GlobalAccessMode::Manage`], where
/// `Manage` additionally covers collection lifecycle and cluster management.
/// Granting it from a scope named `rw` would hand out far more than the name
/// suggests, so callers must opt in with the explicit `qdrant:manage` name to
/// get write access at the global level. Per-collection scopes do express a
/// read/write-without-manage level via [`CollectionAccessMode::ReadWrite`].
///
/// Global scopes take precedence over collection-level scopes. Collection-level
/// scopes are accumulated into a single `Access::Collection` list. Returns
/// `None` when no recognized scope is present.
pub fn access_from_scopes(scopes: &HashSet<String>) -> Option<Access> {
    if scopes.contains("qdrant:manage") {
        return Some(Access::Global(GlobalAccessMode::Manage));
    }
    if scopes.contains("qdrant:read") || scopes.contains("qdrant:r") {
        return Some(Access::Global(GlobalAccessMode::Read));
    }

    let mut collections = Vec::new();
    for scope in scopes {
        let Some(rest) = scope.strip_prefix("qdrant:") else {
            continue;
        };
        let Some((collection, mode)) = rest.rsplit_once(':') else {
            continue;
        };
        if collection.is_empty() {
            continue;
        }
        let access = match mode {
            "rw" => CollectionAccessMode::ReadWrite,
            "r" => CollectionAccessMode::Read,
            _ => continue,
        };
        collections.push(CollectionAccess {
            collection: collection.to_string(),
            access,
            #[expect(deprecated)]
            payload: None,
        });
    }

    if collections.is_empty() {
        return None;
    }

    Some(Access::Collection(CollectionAccessList(collections)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manages_scope_grants_global_manage() {
        let scopes = ["qdrant:manage"].into_iter().map(str::to_owned).collect();
        assert_eq!(
            access_from_scopes(&scopes),
            Some(Access::full("IAM scope qdrant:manage"))
        );
    }

    #[test]
    fn read_scope_grants_global_read() {
        let scopes = ["qdrant:read"].into_iter().map(str::to_owned).collect();
        assert_eq!(
            access_from_scopes(&scopes),
            Some(Access::full_ro("IAM scope qdrant:read"))
        );
    }

    #[test]
    fn r_scope_grants_global_read() {
        let scopes = ["qdrant:r"].into_iter().map(str::to_owned).collect();
        assert_eq!(
            access_from_scopes(&scopes),
            Some(Access::full_ro("IAM scope qdrant:r"))
        );
    }

    #[test]
    fn rw_scope_is_not_a_recognized_global_scope() {
        // There is no global read/write-without-manage access level, so an
        // IAM scope named `rw` must not be silently upgraded to `manage`.
        let scopes = ["qdrant:rw"].into_iter().map(str::to_owned).collect();
        assert_eq!(access_from_scopes(&scopes), None);
    }

    #[test]
    fn global_scope_takes_precedence_over_collection_scopes() {
        let scopes = ["qdrant:r", "qdrant:my_collection:rw"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        assert_eq!(
            access_from_scopes(&scopes),
            Some(Access::full_ro("IAM scope qdrant:r"))
        );
    }

    #[test]
    fn collection_scope_builds_collection_access() {
        let scopes = ["qdrant:products:r", "qdrant:orders:rw"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        let access = access_from_scopes(&scopes).unwrap();
        match access {
            Access::Collection(list) => {
                assert_eq!(list.0.len(), 2);
                // HashSet iteration order is non-deterministic, so look up each
                // collection by name rather than relying on list order.
                let by_name: HashMap<_, _> = list
                    .0
                    .iter()
                    .map(|c| (c.collection.as_str(), c.access))
                    .collect();
                assert_eq!(by_name.get("products"), Some(&CollectionAccessMode::Read));
                assert_eq!(
                    by_name.get("orders"),
                    Some(&CollectionAccessMode::ReadWrite)
                );
            }
            _ => panic!("expected collection access"),
        }
    }

    #[test]
    fn no_recognized_scope_returns_none() {
        let scopes = ["openid", "offline"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        assert_eq!(access_from_scopes(&scopes), None);
    }

    #[test]
    fn empty_scopes_returns_none() {
        let scopes = HashSet::new();
        assert_eq!(access_from_scopes(&scopes), None);
    }

    #[test]
    fn introspection_result_parses_scopes() {
        let result: IntrospectionResult = serde_json::from_str(
            r#"{"active":true,"client_id":"qdrant","scope":"qdrant:manage openid","aud":["qdrant"],"exp":1710000000}"#,
        )
        .unwrap();
        assert!(result.active);
        assert_eq!(result.client_id.as_deref(), Some("qdrant"));
        let scopes = result.scopes();
        assert!(scopes.contains("qdrant:manage"));
        assert!(scopes.contains("openid"));
    }

    #[tokio::test]
    async fn introspects_active_token_and_maps_scopes() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/oauth2/introspect")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"active":true,"client_id":"qdrant","scope":"qdrant:products:rw","sub":"user-1","aud":["qdrant"]}"#,
            )
            .create_async()
            .await;

        let config = IamConfig {
            url: server.url(),
            client_id: "qdrant".to_string(),
            client_secret: "secret".to_string(),
            cache_ttl_sec: 300,
            audience: vec!["qdrant".to_string()],
            timeout_sec: 5,
        };
        let client = IamClient::new(&config);

        let (access, identity) = client.validate("opaque-token").await.unwrap();
        match access {
            Access::Collection(list) => {
                assert_eq!(list.0.len(), 1);
                assert_eq!(list.0[0].collection, "products");
                assert_eq!(list.0[0].access, CollectionAccessMode::ReadWrite);
            }
            _ => panic!("expected collection access"),
        }
        assert_eq!(identity.client_id.as_deref(), Some("qdrant"));
        assert_eq!(identity.sub.as_deref(), Some("user-1"));

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn empty_client_id_is_normalized_to_none() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/oauth2/introspect")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"active":true,"client_id":"","scope":"qdrant:read"}"#)
            .create_async()
            .await;

        let config = IamConfig {
            url: server.url(),
            client_id: "qdrant".to_string(),
            client_secret: "secret".to_string(),
            cache_ttl_sec: 300,
            audience: vec![],
            timeout_sec: 5,
        };
        let client = IamClient::new(&config);

        let (_access, identity) = client.validate("opaque-token").await.unwrap();
        assert_eq!(identity.client_id, None);

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn active_token_with_no_recognized_scope_is_rejected() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/oauth2/introspect")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"active":true,"scope":"openid offline"}"#)
            .create_async()
            .await;

        let config = IamConfig {
            url: server.url(),
            client_id: "qdrant".to_string(),
            client_secret: "secret".to_string(),
            cache_ttl_sec: 300,
            audience: vec![],
            timeout_sec: 5,
        };
        let client = IamClient::new(&config);

        assert!(matches!(
            client.validate("opaque-token").await,
            Err(IamError::NoScope)
        ));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn second_introspection_within_ttl_is_served_from_cache() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/oauth2/introspect")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"active":true,"client_id":"qdrant","scope":"qdrant:read"}"#)
            // Only one HTTP round-trip is expected: the second `introspect`
            // call for the same token must be served from the cache.
            .expect(1)
            .create_async()
            .await;

        let config = IamConfig {
            url: server.url(),
            client_id: "qdrant".to_string(),
            client_secret: "secret".to_string(),
            cache_ttl_sec: 300,
            audience: vec![],
            timeout_sec: 5,
        };
        let client = IamClient::new(&config);

        let first = client.introspect("opaque-token").await.unwrap();
        let second = client.introspect("opaque-token").await.unwrap();
        assert_eq!(first.client_id, second.client_id);

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn cache_entry_does_not_outlive_token_exp() {
        let mut server = mockito::Server::new_async().await;
        // The token expires in 1 second, far sooner than the 300s cache TTL.
        let exp = (Utc::now() + ChronoDuration::seconds(1)).timestamp();
        let mock = server
            .mock("POST", "/oauth2/introspect")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(format!(
                r#"{{"active":true,"client_id":"qdrant","scope":"qdrant:read","exp":{exp}}}"#
            ))
            // One request now, and a second one once the cache entry (capped
            // by `exp`, not by the longer configured TTL) has gone stale.
            .expect(2)
            .create_async()
            .await;

        let config = IamConfig {
            url: server.url(),
            client_id: "qdrant".to_string(),
            client_secret: "secret".to_string(),
            cache_ttl_sec: 300,
            audience: vec![],
            timeout_sec: 5,
        };
        let client = IamClient::new(&config);

        client.introspect("opaque-token").await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;

        client.introspect("opaque-token").await.unwrap();

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn cache_is_bounded_in_size() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/oauth2/introspect")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"active":true,"client_id":"qdrant","scope":"qdrant:read"}"#)
            .expect_at_least(CACHE_MAX_ENTRIES + 1)
            .create_async()
            .await;

        let config = IamConfig {
            url: server.url(),
            client_id: "qdrant".to_string(),
            client_secret: "secret".to_string(),
            cache_ttl_sec: 300,
            audience: vec![],
            timeout_sec: 5,
        };
        let client = IamClient::new(&config);

        for i in 0..=CACHE_MAX_ENTRIES {
            client
                .introspect(&format!("opaque-token-{i}"))
                .await
                .unwrap();
        }

        let cache_len = client.inner.cache.lock().await.len();
        assert!(
            cache_len <= CACHE_MAX_ENTRIES,
            "cache grew past its hard cap: {cache_len}"
        );
    }

    #[tokio::test]
    async fn introspects_inactive_token_as_unauthorized() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/oauth2/introspect")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"active":false,"scope":""}"#)
            .create_async()
            .await;

        let config = IamConfig {
            url: server.url(),
            client_id: "qdrant".to_string(),
            client_secret: "secret".to_string(),
            cache_ttl_sec: 300,
            audience: vec![],
            timeout_sec: 5,
        };
        let client = IamClient::new(&config);

        assert!(matches!(
            client.validate("opaque-token").await,
            Err(IamError::Inactive)
        ));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn rejects_token_with_untrusted_audience() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/oauth2/introspect")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"active":true,"scope":"qdrant:read","aud":["other-service"]}"#)
            .create_async()
            .await;

        let config = IamConfig {
            url: server.url(),
            client_id: "qdrant".to_string(),
            client_secret: "secret".to_string(),
            cache_ttl_sec: 300,
            audience: vec!["qdrant".to_string()],
            timeout_sec: 5,
        };
        let client = IamClient::new(&config);

        assert!(matches!(
            client.validate("opaque-token").await,
            Err(IamError::UntrustedAudience)
        ));
        mock.assert_async().await;
    }
}
