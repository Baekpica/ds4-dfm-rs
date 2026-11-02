use ds4_core::{Backend, DistributedConfig, DistributedRole, Model};

fn routed_quant_bits(model: &Model) {
    let _ = model.routed_quant_bits();
}

#[test]
fn model_exposes_routed_quant_bits() {
    let _ = routed_quant_bits as fn(&Model);
}

fn open_distributed(config: &DistributedConfig) {
    let _ = Model::open_distributed("model.gguf", Backend::Cpu, 0, true, None, None, config);
}

#[test]
fn model_exposes_distributed_oracle_boundary() {
    let config = DistributedConfig {
        role: DistributedRole::Worker,
        layer_start: 21,
        layer_end: u32::MAX,
        has_output: true,
        listen_host: Some("0.0.0.0".into()),
        listen_port: 7100,
        coordinator_host: Some("127.0.0.1".into()),
        coordinator_port: 7000,
        prefill_chunk: 0,
        prefill_window: 0,
        activation_bits: 0,
        debug: false,
    };
    assert_eq!(config.role, DistributedRole::Worker);
    let _ = open_distributed as fn(&DistributedConfig);
}
