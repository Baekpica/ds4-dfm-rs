//! Worker-role launch: assemble path, never HTTP.

use ds4_dist::Role;

pub const WORKER_REQUIRES_MODEL: &str = "ds4-server-rs: --role worker requires -m/--model";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerLaunch {
    Http,
    Worker,
}

impl ServerLaunch {
    /// Apply the allocation policy for the selected launch path before quoting.
    pub fn configure_serving(self, req: &mut ds4_core::ServingRequest) {
        if self == Self::Worker {
            // The assembled worker creates one serial session, with no
            // HTTP batch slabs or continuous scheduler allocation.
            req.max_seqs = ds4_core::MaxSeqs::Off;
            req.lane = ds4_core::LaneMode::Serial;
        }
    }
}

pub fn server_launch(role: Role, has_model: bool) -> Result<ServerLaunch, String> {
    match role {
        Role::Worker => {
            if !has_model {
                return Err(WORKER_REQUIRES_MODEL.to_string());
            }
            Ok(ServerLaunch::Worker)
        }
        Role::None | Role::Coordinator => Ok(ServerLaunch::Http),
    }
}

#[cfg(test)]
mod tests {
    use super::{server_launch, ServerLaunch, WORKER_REQUIRES_MODEL};
    use ds4_dist::{Layers, Role};

    #[test]
    fn worker_quote_has_no_http_banks() {
        use ds4_core::{
            fill_quote_facts, resolve_plan, serving_caps, EngineFacts, MaxSeqs, ModelFamily,
            QuoteHost, ServingRequest, Variant, SHAPE_FLASH,
        };
        let caps = serving_caps(ModelFamily::DeepSeek4, Variant::Flash);
        let mut req = ServingRequest {
            distribution: ds4_core::Distribution::Sliced,
            ..ServingRequest::default()
        };
        server_launch(Role::Worker, true)
            .unwrap()
            .configure_serving(&mut req);
        let mut facts = EngineFacts::default();
        fill_quote_facts(
            &mut facts,
            &req,
            caps,
            Some(SHAPE_FLASH),
            QuoteHost {
                weights_bytes: 0,
                mtp_bytes: 0,
                available_bytes: u64::MAX,
                native_chunk: None,
                vision: false,
            },
        );
        assert_eq!(facts.per_bank_bytes, Some(0));
        assert_eq!(req.max_seqs, MaxSeqs::Off);
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert!(plan.may_listen(), "{:?}", plan.issues);
        assert_eq!(plan.effective.max_seqs, 1);
        let budget = plan.quote.unwrap().total;
        for (available, allowed) in [(budget, true), (budget - 1, false)] {
            facts.host_available_bytes = Some(available);
            let plan = resolve_plan(&req, Some(caps), &facts);
            assert_eq!(plan.may_listen(), allowed, "{:?}", plan.issues);
        }
        let mut http = ServingRequest::default();
        ServerLaunch::Http.configure_serving(&mut http);
        assert_eq!(http.max_seqs, ServingRequest::default().max_seqs);
    }

    #[test]
    fn worker_without_model_requires_m() {
        // Given: --role worker and no -m/--model
        // When: decide launch
        let err = server_launch(Role::Worker, false).unwrap_err();

        // Then: C/shadow error token the bin prints
        assert_eq!(err, WORKER_REQUIRES_MODEL);
        assert!(err.contains("requires -m/--model"));
    }

    #[test]
    fn worker_role_does_not_start_http() {
        // Given: worker role with a model path
        // When: decide launch
        let kind = server_launch(Role::Worker, true).unwrap();

        // Then: HTTP accept loop is not selected
        assert_eq!(kind, ServerLaunch::Worker);
        assert_ne!(kind, ServerLaunch::Http);
        assert_eq!(
            server_launch(Role::None, false).unwrap(),
            ServerLaunch::Http
        );
        assert_eq!(
            server_launch(Role::Coordinator, true).unwrap(),
            ServerLaunch::Http
        );
    }

    #[test]
    fn worker_assemble_uses_bound_listen_port() {
        // Given: worker layers and an ephemeral data listener
        let layers = Layers {
            start: 20,
            end: 20,
            has_output: true,
            set: true,
        };
        let meta = ds4_dist::slice_meta(7, 43, 129_280, 4096, 7168, &layers);
        let (_listener, port) = ds4_dist::open_data_listener(Some("127.0.0.1"), 0).unwrap();

        // When: plan HELLO via the dist assemble helper
        let plan = ds4_dist::worker_plan(&meta, 2, u32::from(port), "deepseek4");

        // Then: HELLO carries the bound nonzero data port
        assert_ne!(plan.hello.listen_port, 0);
        assert_eq!(plan.hello.listen_port, u32::from(port));
        assert_eq!(plan.hello.layer_start, 20);
        assert_eq!(plan.hello.has_output, 1);
        assert_eq!(plan.model_name, "deepseek4");
    }
}
