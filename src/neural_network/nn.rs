use super::super::helpers;
use super::activation_functions::ActivationFunctions;
use super::genes::{ConnectionGene, NodeGene, NodeType};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::Rng;
use std::collections::HashMap;

const MUTATE_METHOD: [&str; 7] = ["weight", "bias", "node", "conn", "squash", "rnode", "rconn"];
const MUTATE_METHOD_WEIGHTS: [u32; 7] = [1, 1, 1, 1, 1, 1, 1];

#[derive(Debug, PartialEq)]
pub struct NN {
    pub node_genes_network: Vec<Vec<u32>>,
    pub node_genes: HashMap<u32, NodeGene>,
    pub connection_genes: Vec<ConnectionGene>,
    pub fitness: f32,
    pub next_id: u32,
}

impl NN {
    pub fn new(input_layer: Vec<NodeGene>, output_layer: Vec<NodeGene>) -> Self {
        // Connects every input_layer node to the output_layer node
        let mut connection_genes = vec![];
        let mut next_id = 0;

        for (i, _) in input_layer.clone().into_iter().enumerate() {
            if input_layer[i].node_type != NodeType::Input {
                continue;
            }
            for (j, _) in output_layer.clone().into_iter().enumerate() {
                if output_layer[j].node_type != NodeType::Output {
                    continue;
                }
                connection_genes.push(ConnectionGene::new(
                    input_layer[i].id,
                    output_layer[j].id,
                    helpers::std0(),
                    -1,
                ));
                next_id += 1;
            }
        }

        let mut node_genes = HashMap::new();
        let mut node_genes_network = vec![vec![], vec![]];

        for node in input_layer {
            node_genes.insert(node.id, node);
            node_genes_network[0].push(node.id);
        }
        for node in output_layer {
            node_genes.insert(node.id, node);
            node_genes_network[1].push(node.id);
        }

        NN {
            node_genes_network: node_genes_network,
            node_genes: node_genes,
            connection_genes: connection_genes,
            fitness: 0.0,
            next_id: next_id,
        }
    }

    // Tried with derive(Copy, Clone)...
    pub fn clone(nn: &NN) -> NN {
        let mut node_genes_network = vec![];

        for layer in nn.node_genes_network.clone() {
            let mut network_layer = vec![];
            for node in layer {
                network_layer.push(node.clone());
            }
            node_genes_network.push(network_layer)
        }

        let mut connection_genes = vec![];
        for conn in nn.connection_genes.clone() {
            connection_genes.push(conn);
        }

        NN {
            node_genes_network: node_genes_network,
            node_genes: nn.node_genes.clone(),
            connection_genes: connection_genes,
            fitness: nn.fitness,
            next_id: nn.next_id,
        }
    }

    fn get_output(&mut self) -> usize {
        // return the index of the output node with highest value
        for conn in &self.connection_genes {
            let from_value = match self.node_genes.get_mut(&conn.from) {
                Some(node) => node.get_output(),
                None => continue,
            };

            match self.node_genes.get_mut(&conn.to) {
                Some(node) => node.add_value(from_value),
                None => continue,
            }
        }

        let mut cur: (f32, usize) = (0.0, 0);

        let outputs = match self.node_genes_network.last() {
            Some(node) => node,
            e => panic!("{:?}", e),
        };

        for (i, node_id) in outputs.iter().enumerate() {
            match self.node_genes.get_mut(&node_id) {
                Some(node) => {
                    let value = node.get_output();
                    // set cur if it is the first or get the requirements
                    if value > cur.0 || i == 0 {
                        cur.0 = value;
                        cur.1 = i;
                    }
                }
                _ => println!("This should give a value..."),
            };
        }
        cur.1
    }

    fn reset_gene_value(&mut self) {
        // want to reset the value (set it to 0) when you have got an ouput
        for row in &self.node_genes_network {
            for key in row {
                match self.node_genes.get_mut(key) {
                    Some(node) => node.value = 0.0,
                    _ => (),
                }
            }
        }
    }

    pub fn update(&mut self, input_values: Vec<f32>) -> usize {
        for (i, key) in self.node_genes_network[0].iter().enumerate() {
            match self.node_genes.get_mut(&key) {
                Some(node) => node.add_value(input_values[i]),
                _ => continue,
            }
        }
        let res = self.get_output();

        self.reset_gene_value();
        res
    }

    pub fn calculate_fitness(&self) -> f32 {
        let mut fitenss = self.fitness;
        self.node_genes_network
            .iter()
            .for_each(|f| fitenss -= f.len() as f32);

        fitenss -= self.connection_genes.len() as f32;
        fitenss
    }

    pub fn reward(&mut self, reward: f32) {
        // this funciton add a reward (or a penalty) to the fitness
        self.fitness += reward;
    }

    fn get_next_id(&mut self) -> u32 {
        self.next_id += 1;
        self.next_id
    }

    pub fn mutate(&mut self) {
        // This adds a random mutation (e.g. extra node, or adjusted weight)
        let mut rng = thread_rng();
        let dist = WeightedIndex::new(&MUTATE_METHOD_WEIGHTS).unwrap();
        let res = MUTATE_METHOD[dist.sample(&mut rng)];
        match res {
            "weight" => self.mutate_weight(),
            "bias" => self.mutate_bias(),
            "node" => self.mutate_new_node(),
            "conn" => self.mutate_new_connection(),
            "squash" => self.mutate_squash(),
            "rnode" => self.mutate_remove_node(),
            "rconn" => self.mutate_remove_connection(),
            _ => (),
        }
    }

    fn mutate_weight(&mut self) {
        // This method takes a random connection and changes its weight
        match self.connection_genes.choose_mut(&mut rand::thread_rng()) {
            Some(conn) => conn.weight += helpers::lib::std0(),
            _ => (),
        }
    }

    fn mutate_bias(&mut self) {
        // This method takes a random node and changes its bias
        match self.node_genes_network.choose_mut(&mut rand::thread_rng()) {
            Some(layer) => match layer.choose_mut(&mut rand::thread_rng()) {
                Some(key) => match self.node_genes.get_mut(key) {
                    Some(node) => node.bias += helpers::lib::std0(),
                    e => panic!("{:?}", e),
                },
                e => panic!("{:?}", e),
            },
            e => panic!("{:?}", e),
        };
    }

    fn mutate_new_node(&mut self) {
        // When you add a extra node, it reduces the fitness
        self.fitness -= 1.0;
        let mut rng = rand::thread_rng();
        let upper_limit = self.node_genes_network.len() - 2;
        let layer_no = rng.gen_range(0..=upper_limit);

        let from_layer;
        let to_layer;

        let id = self.get_next_id();
        let node = NodeGene::new(
            id,
            NodeType::Hidden,
            helpers::lib::std0(),
            ActivationFunctions::get_random(),
            0.0,
        );
        self.node_genes.insert(id, node);

        let mut check_before = vec![];

        if layer_no == 0 {
            from_layer = upper_limit;
            to_layer = upper_limit + 1;

            let new_layer = vec![id];
            self.node_genes_network.insert(upper_limit, new_layer);

            for (i, layer) in self.node_genes_network.iter().enumerate() {
                if i == upper_limit {
                    break;
                }
                for row in layer {
                    check_before.push(row);
                }
            }
        } else {
            from_layer = layer_no - 1;
            to_layer = layer_no + 1;
            self.node_genes_network[layer_no].push(id);
            for (i, layer) in self.node_genes_network.iter().enumerate() {
                if i == layer_no {
                    break;
                }
                for row in layer {
                    check_before.push(row);
                }
            }
        }

        let mut rng = rand::thread_rng();
        let from_node_id = self.node_genes_network[from_layer]
            .choose(&mut rng)
            .unwrap();
        let to_node_id = self.node_genes_network[to_layer].choose(&mut rng).unwrap();
        let conn1 = ConnectionGene::new(*from_node_id, id, helpers::lib::std0(), -1);
        let conn2 = ConnectionGene::new(id, *to_node_id, helpers::lib::std0(), -1);

        let mut inserted_conn1 = false;

        for i in 0..self.connection_genes.len() {
            if self.connection_genes[i].from == *from_node_id {
                self.connection_genes.insert(i, conn1);
                inserted_conn1 = true;
                break;
            }
        }

        if !inserted_conn1 {
            self.connection_genes.push(conn1);
        }

        for i in 0..self.connection_genes.len() {
            if !check_before.contains(&&id) {
                self.connection_genes.insert(i, conn2);
                break;
            }
        }
    }

    fn mutate_new_connection(&mut self) {
        // if all connections are made
        let mut count = 0;
        loop {
            if count > self.node_genes_network.len() {
                // even if it has not checked every possibiliy, the calculation
                // is not worth it
                break;
            }
            let mut made_new_conn = true;
            let mut rng = rand::thread_rng();
            let from_idx = rng.gen_range(0..self.node_genes_network.len() - 1);
            let to_idx = rng.gen_range(from_idx + 1..self.node_genes_network.len());

            let from_id = self.node_genes_network[from_idx].choose(&mut rng).unwrap();
            let to_id = self.node_genes_network[to_idx].choose(&mut rng).unwrap();

            // check if already exists
            for conn in &self.connection_genes {
                if conn.from == *from_id && conn.to == *to_id {
                    made_new_conn = false;
                    break;
                }
            }

            count += 1;
            if !made_new_conn {
                continue;
            }

            let mut inserted = false;

            let conn = ConnectionGene::new(*from_id, *to_id, helpers::lib::std0(), -1);
            for i in 0..self.connection_genes.len() {
                if self.connection_genes[i].from == *from_id {
                    self.connection_genes.insert(i, conn);
                    inserted = true;
                    break;
                }
            }

            if !inserted {
                self.connection_genes.push(conn);
            }
            break;
        }
    }

    fn mutate_squash(&mut self) {
        match self.node_genes_network.choose_mut(&mut rand::thread_rng()) {
            Some(layer) => match layer.choose_mut(&mut rand::thread_rng()) {
                Some(key) => match self.node_genes.get_mut(key) {
                    Some(node) => node.squash = ActivationFunctions::get_random(),
                    e => panic!("{:?}", e),
                },
                e => panic!("{:?}", e),
            },
            e => panic!("{:?}", e),
        };
    }

    fn mutate_remove_node(&mut self) {
        let mut rng = rand::thread_rng();
        if self.node_genes_network.len() == 2 {
            // can only remove hidden layers
            return;
        }
        let layer_idx = rng.gen_range(1..self.node_genes_network.len() - 1);
        let node_id: u32;

        if self.node_genes_network[layer_idx].len() == 1 {
            node_id = self.node_genes_network[layer_idx][0];
            self.node_genes_network.remove(layer_idx);
        } else {
            let node_idx = rng.gen_range(0..self.node_genes_network[layer_idx].len());
            node_id = self.node_genes_network[layer_idx][node_idx];
            self.node_genes_network[layer_idx].remove(node_idx);
        }

        self.node_genes.remove(&node_id);

        let mut new_connection_genes: Vec<ConnectionGene> = vec![];
        self.connection_genes.iter().for_each(|node| {
            if node.from != node_id && node.to != node_id {
                new_connection_genes.push(*node)
            }
        });

        self.connection_genes = new_connection_genes;
    }

    fn mutate_remove_connection(&mut self) {
        if self.connection_genes.len() == 0 {
            return;
        }
        let mut rng = rand::thread_rng();
        self.connection_genes
            .remove(rng.gen_range(0..self.connection_genes.len()));
    }
}

#[cfg(test)]
mod test_neural_network_mutations {
    use super::*;
    #[test]
    fn test_mutate_weight() {
        let node_genes_network = vec![];
        let node_genes = HashMap::new();

        let connection_genes = vec![ConnectionGene {
            from: 0,
            to: 0,
            weight: 0.0,
            gater: -1,
        }];

        let mut nn = NN {
            node_genes_network: node_genes_network,
            node_genes: node_genes,
            connection_genes: connection_genes,
            fitness: 0.0,
            next_id: 0,
        };

        nn.mutate_weight();

        assert_ne!(nn.connection_genes[0].weight, 0.0);
    }

    #[test]
    fn test_mutate_bias() {
        let node_genes_network = vec![vec![0]];
        let node = NodeGene::new(0, NodeType::Hidden, 0.0, ActivationFunctions::None, 0.0);
        let mut node_genes = HashMap::new();
        node_genes.insert(0, node);

        let mut nn = NN {
            node_genes_network: node_genes_network,
            node_genes: node_genes,
            connection_genes: vec![],
            fitness: 0.0,
            next_id: 0,
        };

        nn.mutate_bias();
        match nn.node_genes.get(&0) {
            Some(node) => assert_ne!(node.bias, 0.0),
            e => panic!("{:?}", e),
        }
    }

    #[test]
    fn test_mutate_new_node() {
        let input_nodes = vec![NodeGene::new(
            0,
            NodeType::Input,
            0.0,
            ActivationFunctions::None,
            0.0,
        )];
        let output_nodes = vec![NodeGene::new(
            1,
            NodeType::Output,
            0.0,
            ActivationFunctions::None,
            0.0,
        )];
        let mut nn = NN::new(input_nodes, output_nodes);
        nn.mutate_new_node();

        assert_eq!(nn.connection_genes.len(), 3);
        assert_eq!(nn.node_genes_network.len(), 3);
        assert_eq!(nn.node_genes_network[1].len(), 1);

        match nn.node_genes.get(&2) {
            Some(node) => assert_eq!(node.id, 2),
            e => panic!("{:?}", e),
        }
    }

    #[test]
    fn test_mutate_new_connection() {
        let input_nodes = vec![NodeGene::new(
            0,
            NodeType::Input,
            0.0,
            ActivationFunctions::None,
            0.0,
        )];
        let output_nodes = vec![NodeGene::new(
            1,
            NodeType::Output,
            0.0,
            ActivationFunctions::None,
            0.0,
        )];
        let mut nn = NN::new(input_nodes, output_nodes);
        nn.mutate_new_connection();

        // it should not add if exists
        assert_eq!(nn.connection_genes.len(), 1);

        nn.connection_genes.pop();

        nn.mutate_new_connection();

        assert_eq!(nn.connection_genes.len(), 1);
    }

    #[test]
    fn test_mutate_squash() {
        let input_nodes = vec![NodeGene::new(
            0,
            NodeType::Input,
            0.0,
            ActivationFunctions::None,
            0.0,
        )];
        let output_nodes = vec![NodeGene::new(
            1,
            NodeType::Output,
            0.0,
            ActivationFunctions::None,
            0.0,
        )];

        let mut nn = NN::new(input_nodes, output_nodes);
        nn.mutate_squash();

        match nn.node_genes.get(&0) {
            Some(node1) => match nn.node_genes.get(&1) {
                Some(node2) => {
                    assert!(
                        node1.squash != ActivationFunctions::None
                            || node2.squash != ActivationFunctions::None
                    )
                }
                e => panic!("{:?}", e),
            },
            e => panic!("{:?}", e),
        }
    }

    #[test]
    fn test_mutate_remove_node() {
        let input_nodes = vec![NodeGene::new(
            0,
            NodeType::Input,
            0.0,
            ActivationFunctions::None,
            0.0,
        )];
        let output_nodes = vec![NodeGene::new(
            1,
            NodeType::Output,
            0.0,
            ActivationFunctions::None,
            0.0,
        )];

        let hidden_node = NodeGene::new(2, NodeType::Hidden, 0.0, ActivationFunctions::None, 0.0);

        let mut nn = NN::new(input_nodes, output_nodes);
        nn.node_genes_network.insert(1, vec![2]);
        nn.node_genes.insert(2, hidden_node);

        let conn1 = ConnectionGene::new(0, 2, 0.0, -1);
        let conn2 = ConnectionGene::new(2, 1, 0.0, -1);
        nn.connection_genes.push(conn1);
        nn.connection_genes.push(conn2);

        nn.mutate_remove_node();
        assert_eq!(nn.connection_genes.len(), 1);
        assert_eq!(nn.node_genes_network.len(), 2);
        match nn.node_genes.get(&2) {
            Some(node) => panic!("Should not give: {:?}", node),
            _ => (),
        }
    }

    #[test]
    fn test_mutate_remove_connection() {
        let input_nodes = vec![NodeGene::new(
            0,
            NodeType::Input,
            0.0,
            ActivationFunctions::None,
            0.0,
        )];
        let output_nodes = vec![NodeGene::new(
            1,
            NodeType::Output,
            0.0,
            ActivationFunctions::None,
            0.0,
        )];

        let mut nn = NN::new(input_nodes, output_nodes);

        nn.mutate_remove_connection();
        assert_eq!(nn.connection_genes.len(), 0);
    }
}

#[cfg(test)]
mod test_neural_network {
    use super::*;

    #[test]
    fn test_nn_new() {
        let input = vec![
            NodeGene::new(0, NodeType::Input, 0.0, ActivationFunctions::None, 0.0),
            NodeGene::new(1, NodeType::Input, 0.0, ActivationFunctions::None, 0.0),
            NodeGene::new(2, NodeType::Hidden, 0.0, ActivationFunctions::None, 0.0),
        ];

        let output = vec![
            NodeGene::new(3, NodeType::Output, 0.0, ActivationFunctions::None, 0.0),
            NodeGene::new(4, NodeType::Hidden, 0.0, ActivationFunctions::None, 0.0),
            NodeGene::new(5, NodeType::Output, 0.0, ActivationFunctions::None, 0.0),
        ];

        let nn = NN::new(input, output);

        println!("{:?}", nn.connection_genes);

        assert_eq!(nn.connection_genes[0].to, 3);
        assert_eq!(nn.connection_genes[0].from, 0);
        assert_eq!(nn.connection_genes[1].to, 5);
        assert_eq!(nn.connection_genes[1].from, 0);
        assert_eq!(nn.connection_genes[2].to, 3);
        assert_eq!(nn.connection_genes[2].from, 1);
        assert_eq!(nn.connection_genes[3].to, 5);
        assert_eq!(nn.connection_genes[3].from, 1);
    }

    #[test]
    // test the clone method
    fn test_nn_clone() {
        let input_layer = vec![NodeGene::new(
            0,
            NodeType::Input,
            0.0,
            ActivationFunctions::None,
            0.0,
        )];

        let output_layer = vec![NodeGene::new(
            1,
            NodeType::Output,
            0.0,
            ActivationFunctions::None,
            0.0,
        )];

        let nn = NN::new(input_layer, output_layer);

        let nn_clone = NN::clone(&nn);

        assert_eq!(nn.node_genes_network[0], nn_clone.node_genes_network[0]);
        assert_eq!(nn.node_genes_network[1], nn_clone.node_genes_network[1]);
        assert_eq!(nn.connection_genes[0], nn_clone.connection_genes[0]);
        assert_eq!(nn.fitness, nn_clone.fitness);
    }

    #[test]
    fn test_get_output() {
        let input_layer = vec![
            NodeGene::new(0, NodeType::Input, 0.0, ActivationFunctions::None, 1.0),
            NodeGene::new(1, NodeType::Input, 0.0, ActivationFunctions::None, 2.0),
        ];

        let output_layer = vec![
            NodeGene::new(2, NodeType::Output, 0.0, ActivationFunctions::None, 0.0),
            NodeGene::new(3, NodeType::Output, 0.0, ActivationFunctions::None, 0.0),
        ];

        let hidden_layer = vec![NodeGene::new(
            4,
            NodeType::Output,
            1.0,
            ActivationFunctions::None,
            1.0,
        )];

        let node_genes_network = vec![vec![0, 1], vec![4], vec![2, 3]];

        let mut node_genes = HashMap::new();
        node_genes.insert(0, input_layer[0]);
        node_genes.insert(1, input_layer[1]);
        node_genes.insert(2, output_layer[0]);
        node_genes.insert(3, output_layer[1]);
        node_genes.insert(4, hidden_layer[0]);

        let connection_genes = vec![
            ConnectionGene::new(0, 2, 1.0, -1),
            ConnectionGene::new(0, 3, 1.0, -1),
            ConnectionGene::new(0, 4, 2.0, -1),
            ConnectionGene::new(1, 2, 1.0, -1),
            ConnectionGene::new(1, 3, 1.0, -1),
            ConnectionGene::new(4, 3, 2.0, -1),
        ];

        let mut nn = NN {
            node_genes_network: node_genes_network,
            node_genes: node_genes,
            connection_genes: connection_genes,
            fitness: 0.0,
            next_id: 0,
        };

        let res = nn.get_output();

        assert_eq!(res, 1);

        match nn.node_genes.get(&2) {
            Some(node) => assert_eq!(node.value, 3.0),
            None => panic!("It should return a value"),
        };

        match nn.node_genes.get(&3) {
            Some(node) => assert_eq!(node.value, 6.0),
            None => panic!("It should return a value"),
        };
    }

    #[test]
    fn test_calculate_fitness() {
        let node_genes_network = vec![vec![0, 1], vec![3], vec![2]];

        let nn = NN {
            node_genes_network: node_genes_network,
            node_genes: HashMap::new(),
            connection_genes: vec![ConnectionGene::new(1, 0, 0.0, -1)],
            fitness: 10.0,
            next_id: 0,
        };

        // For now this should be ok, but should make a better fitness function
        assert_eq!(nn.calculate_fitness(), 5.0);
    }

    #[test]
    fn test_update() {
        let input = vec![
            NodeGene::new(0, NodeType::Input, 0.0, ActivationFunctions::None, 0.0),
            NodeGene::new(1, NodeType::Input, 0.0, ActivationFunctions::None, 0.0),
        ];

        let output = vec![
            NodeGene::new(2, NodeType::Output, 0.0, ActivationFunctions::None, 0.0),
            NodeGene::new(3, NodeType::Output, 0.0, ActivationFunctions::None, 0.0),
        ];

        let hidden = NodeGene::new(4, NodeType::Hidden, 1.0, ActivationFunctions::None, 1.0);

        let con1 = ConnectionGene::new(0, 4, 1.0, -1);
        let con2 = ConnectionGene::new(4, 2, 1.0, -1);

        let mut nn = NN::new(input, output);
        nn.node_genes_network.insert(1, vec![4]);
        nn.node_genes.insert(4, hidden);
        nn.connection_genes.insert(2, con1);
        nn.connection_genes.push(con2);
        let res = nn.update(vec![1.0, 2.0]);

        assert_eq!(res, 0);

        // the node value should be 0.0 bc it resets it after it has given an output
        match nn.node_genes.get(&2) {
            Some(node) => assert_eq!(node.value, 0.0),
            e => println!("{:?}", e),
        }

        match nn.node_genes.get(&3) {
            Some(node) => assert_eq!(node.value, 0.0),
            e => println!("{:?}", e),
        }
    }

    #[test]
    fn test_reset_gene_value() {
        let node = NodeGene::new(0, NodeType::Input, 0.0, ActivationFunctions::None, 1.0);

        let mut nn = NN::new(vec![node], vec![]);
        nn.reset_gene_value();
        match nn.node_genes.get(&0) {
            Some(node) => assert_eq!(node.value, 0.0),
            e => panic!("{:?}", e),
        }
    }
}
