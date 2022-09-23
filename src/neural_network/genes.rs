use super::activation_functions::ActivationFunctions;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum NodeType {
    Input,
    Hidden,
    Output,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct NodeGene {
    pub id: u32,
    pub node_type: NodeType,
    pub bias: f32,
    pub squash: ActivationFunctions,
    pub value: f32,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct ConnectionGene {
    pub from: u32, // from node
    pub to: u32,   // to node
    pub weight: f32,
    pub gater: i32,
}

impl ConnectionGene {
    pub fn new(from: u32, to: u32, weight: f32, gater: i32) -> Self {
        ConnectionGene {
            from: from,
            to: to,
            weight: weight,
            gater: gater,
        }
    }
}

impl NodeGene {
    pub fn new(
        id: u32,
        node_type: NodeType,
        bias: f32,
        squash: ActivationFunctions,
        value: f32,
    ) -> Self {
        NodeGene {
            id: id,
            node_type: node_type,
            bias: bias,
            squash: squash,
            value: value,
        }
    }

    fn calculate_value(&mut self, value: f32) -> f32 {
        ActivationFunctions::get_output(self.squash, value)
    }

    pub fn get_output(&mut self) -> f32 {
        self.calculate_value(self.value + self.bias)
    }

    pub fn add_value(&mut self, value: f32) {
        self.value += value;
    }
}

#[cfg(test)]
mod test_genes {
    use super::*;
    #[test]
    fn test_calculate_value() {
        let mut node = NodeGene::new(0, NodeType::Hidden, 1.0, ActivationFunctions::None, 0.0);

        assert_eq!(node.calculate_value(1.0), 1.0);
    }
}
