use super::super::helpers;
use super::activation_functions::ActivationFunctions;
use super::genes::{NodeGene, NodeType};
use super::nn::NN;
use rand::Rng;

pub struct Environment {
    pub species: Vec<NN>,
    pub generation: u32,
}

impl Environment {
    pub fn new(num_of_species: u32, num_input_layer: u32, num_output_layer: u32) -> Self {
        let mut species = vec![];
        let mut input_layer = vec![];
        let mut id = 0;

        // Want to set the id for output layer first so it can reference the index
        let mut output_layer = vec![];
        for _ in 0..num_output_layer {
            output_layer.push(NodeGene::new(
                id,
                NodeType::Output,
                helpers::lib::std0(),
                ActivationFunctions::Identity,
                0.0,
            ));
            id += 1;
        }

        for _ in 0..num_input_layer {
            input_layer.push(NodeGene::new(
                id,
                NodeType::Input,
                helpers::lib::std0(),
                ActivationFunctions::Identity,
                0.0,
            ));
            id += 1;
        }

        for _ in 0..num_of_species {
            species.push(NN::new(input_layer.clone(), output_layer.clone()))
        }

        Environment {
            species: species,
            generation: 1,
        }
    }

    pub fn next_generation(&mut self, local_maximum: bool) {
        self.generation += 1;
        let mut nn: NN;
        if local_maximum {
            // if it has a local maximum, then get a random specie
            // the local maximum is desided by the user/variable
            let mut rng = rand::thread_rng();
            let num = rng.gen_range(0..self.species.len());
            nn = NN::clone(&self.species[num]);
            let i = rng.gen_range(1..4);
            for _ in 0..i {
                nn.mutate();
            }
        } else {
            nn = self.get_best_specie();
        }
        // want to set the fitness to 0 again when new generation
        nn.fitness = 0.0;
        let mut new_species = vec![];
        let mut rng = rand::thread_rng();
        for _ in 0..self.species.len() - 1 {
            let mut new_nn = NN::clone(&nn);
            // mutate a random number of times (up to 3)
            let i = rng.gen_range(1..3);
            for _ in 0..i {
                new_nn.mutate();
            }
            new_species.push(new_nn);
        }
        new_species.push(NN::clone(&nn));

        self.species = new_species;
    }

    pub fn get_best_specie(&mut self) -> NN {
        // NOTE: Now is the fittest only used

        // find the fittest specie
        // it only checks the fitness
        // the fitness should be a function that takes number of nodes and connections in
        // consideration

        let mut fittest: &NN = &NN::clone(&self.species[0]);
        let mut next_fittest: &NN = &NN::clone(&self.species[0]);

        for specie in &self.species {
            // check if it is fitter than the fittest
            // if it is, push fittest to next fittest
            if specie.calculate_fitness() > fittest.calculate_fitness() {
                next_fittest = fittest;
                fittest = specie;
                continue;
            }

            // if not fitter than the fittest, compare to the next fittest

            if specie.calculate_fitness() > next_fittest.calculate_fitness() {
                next_fittest = specie;
                continue;
            }
        }

        // if two species has the same fitness, merge the two species
        // NN::merge(fittest, next_fittest)
        NN::clone(fittest)
    }
}

#[cfg(test)]
mod test_environment {
    use super::*;

    #[test]
    // test the environment::new()
    // the length should be equal to the num_of_species
    fn test_environment_new() {
        let num_of_species = 2;

        let env = Environment::new(num_of_species, 2, 2);

        assert_eq!(env.species.len(), 2);
    }

    #[test]
    // test pick the best species
    // choose the best, if two with the same fitness level, merge these two
    // if multiple with same fitness, choose the two with least nodes and merge them
    fn test_best_species() {
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

        let mut nn1 = NN::new(input_layer, output_layer);
        let mut nn2 = NN::clone(&nn1);
        let mut nn3 = NN::clone(&nn1);
        let mut nn4 = NN::clone(&nn1);
        let mut nn5 = NN::clone(&nn1);

        nn1.fitness = 1.0;
        nn2.fitness = 0.0;

        let mut env = Environment::new(0, 0, 0);
        env.species.push(nn1);
        env.species.push(nn2);

        assert_eq!(env.get_best_specie().fitness, 1.0);

        nn3.fitness = 2.0;
        nn4.fitness = 2.0;
        nn5.fitness = 0.0;

        let mut env = Environment::new(0, 0, 0);
        env.species.push(nn5);
        env.species.push(nn3);
        env.species.push(nn4);

        // this should be equal nn3 or nn4 becaus they have the same connection_nodes and same fitness
        assert_eq!(env.get_best_specie().fitness, 2.0);

        // TODO: add a test for species with same fintness but different complexity
        // want to choose the specie with least nodes!
    }
}
