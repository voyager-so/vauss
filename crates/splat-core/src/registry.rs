use crate::{Backend, ShotTools};
use anyhow::{Context, Result};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Default)]
pub struct Registry {
    backends: BTreeMap<String, Arc<dyn Backend>>,
    pub(crate) shots: Option<Arc<dyn ShotTools>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn register_backend(&mut self, backend: Arc<dyn Backend>) {
        self.backends.insert(backend.name().into(), backend);
    }
    pub fn set_shots(&mut self, shots: Arc<dyn ShotTools>) {
        self.shots = Some(shots);
    }
    pub fn backend(&self, name: &str) -> Result<&dyn Backend> {
        self.backends
            .get(name)
            .map(Arc::as_ref)
            .with_context(|| format!("Backend {name:?} is not registered"))
    }
}
