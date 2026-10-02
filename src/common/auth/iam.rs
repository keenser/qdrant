use std::collections::{HashMap, HashSet};
use std::time::Duration;

use chrono::{Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use storage::rbac::{
    Access, CollectionAccess, CollectionAccessList, CollectionAccessMode, GlobalAccessMode,
};
use tokio::sync::Mutex;

use crate::settings::IamConfig;

/// A cached introspection result.
///
/// The whole entry is stored together with the point in time it was cached, so
/// a single `Mutex<HashMap<String, CachedIntrospection>>` gives us both the
/// cache TTL and per-token results.
struct CachedIntrospection {
    result: IntrospectionResult,
    cached_at: chrono::DateTime<Utc>,
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
    /// The OAuth2 client that requested the token.
    pub client_id: String,
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
    cache_ttl: Duration,
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
}

/// Encode key/value pairs into an `application/x-www-form-urlencoded` body.
fn urlencoded(pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .map(|(k, v)| format!("{}={}", percent_encode(k), percent_encode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

fn percent_encode(s: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(s.len());
    for byte in s.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char)
            }
            _ => {
                out.push('%');
                out.push(HEX[(byte >> 4) as usize] as char);
                out.push(HEX[(byte & 0x0f) as usize] as char);
            }
        }
    }
    out
}

impl IamClient {
    /// Create a new IAM client from the given configuration.
    pub fn new(config: &IamConfig) -> Self {
        let introspection_url = format!("{}/oauth2/introspect", config.url.trim_end_matches('/'));
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_sec))
            .build()
            .expect("Failed to build IAM HTTP client");

        Self {
            inner: std::sync::Arc::new(IamClientInner {
                introspection_url,
                client_id: config.client_id.clone(),
                client_secret: config.client_secret.clone(),
                cache_ttl: Duration::from_secs(config.cache_ttl_sec),
                audience: config.audience.iter().cloned().collect(),
                http,
                cache: Mutex::new(Default::default()),
            }),
        }
    }

    /// Introspect an opaque token, using the in-memory cache when possible.
    ///
    /// Returns the introspection result. `None` is returned when the token is
    /// not active or its audience is not trusted.
    pub async fn introspect(&self, token: &str) -> Result<IntrospectionResult, IamError> {
        // Fast path: serve from cache when a non-expired entry exists.
        {
            let cache = self.inner.cache.lock().await;
            if let Some(entry) = cache.get(token)
                && entry.cached_at + ChronoDuration::from_std(self.inner.cache_ttl).unwrap()
                    >= Utc::now()
            {
                return Ok(entry.result.clone());
            }
        }

        let result = self.introspect_remote(token).await?;

        // Cache only positive results; negative results are cheap to recompute
        // and may flip back to active at any moment.
        if result.active {
            let mut cache = self.inner.cache.lock().await;
            cache.insert(
                token.to_string(),
                CachedIntrospection {
                    result: result.clone(),
                    cached_at: Utc::now(),
                },
            );
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

        let access = access_from_scopes(&result.scopes()).ok_or(IamError::Inactive)?;
        let identity = IamIdentity {
            client_id: result.client_id.unwrap_or_default(),
            sub: result.sub.clone(),
        };

        Ok((access, identity))
    }

    async fn introspect_remote(&self, token: &str) -> Result<IntrospectionResult, IamError> {
        let body = urlencoded(&[("token", token), ("token_type_hint", "access_token")]);

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

/// Build a Qdrant [`Access`] from a set of IAM scopes.
///
/// Supported scope conventions:
/// - `qdrant:manage`                  -> global manage (full) access
/// - `qdrant:read`                    -> global read-only access
/// - `qdrant:rw`                      -> read/write access to all collections
/// - `qdrant:r`                       -> read-only access to all collections
/// - `qdrant:<collection>:rw`         -> read/write access to a specific collection
/// - `qdrant:<collection>:r`          -> read-only access to a specific collection
///
/// Global scopes take precedence over collection-level scopes. Collection-level
/// scopes are accumulated into a single `Access::Collection` list. Returns
/// `None` when no recognized scope is present.
pub fn access_from_scopes(scopes: &HashSet<String>) -> Option<Access> {
    if scopes.contains("qdrant:manage") {
        return Some(Access::Global(GlobalAccessMode::Manage));
    }
    if scopes.contains("qdrant:read") {
        return Some(Access::Global(GlobalAccessMode::Read));
    }
    if scopes.contains("qdrant:rw") {
        return Some(Access::Global(GlobalAccessMode::Manage));
    }
    if scopes.contains("qdrant:r") {
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
    fn rw_scope_grants_global_manage() {
        let scopes = ["qdrant:rw"].into_iter().map(str::to_owned).collect();
        assert_eq!(
            access_from_scopes(&scopes),
            Some(Access::full("IAM scope qdrant:rw"))
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
        assert_eq!(identity.client_id, "qdrant");
        assert_eq!(identity.sub.as_deref(), Some("user-1"));

        mock.assert_async().await;
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
