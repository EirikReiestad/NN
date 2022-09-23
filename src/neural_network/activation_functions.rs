use libm;
use rand::seq::SliceRandom;

#[allow(dead_code)]
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum ActivationFunctions {
    None,
    Logistics,
    HyperbolicTangent,
    Identity,
    BinaryStep,
    ReLU,
    Softsign,
    Gaussian,
    Sinusiodial,
    BentIdentity,
    SELU,
}

impl ActivationFunctions {
    pub fn get_random() -> ActivationFunctions {
        let mut rng = rand::thread_rng();
        let choices = [
            ActivationFunctions::Logistics,
            ActivationFunctions::HyperbolicTangent,
            ActivationFunctions::Identity,
            ActivationFunctions::BinaryStep,
            ActivationFunctions::ReLU,
            ActivationFunctions::Softsign,
            ActivationFunctions::Gaussian,
            ActivationFunctions::Sinusiodial,
            ActivationFunctions::BentIdentity,
        ];
        *choices.choose(&mut rng).unwrap()
    }

    fn logistic_function(value: f32) -> f32 {
        1.0 / (1.0 + libm::exp(-value as f64)) as f32
    }

    fn hyperbolic_tangent_function(value: f32) -> f32 {
        let value: f64 = value as f64;
        (libm::exp(value) - libm::exp(-value)) as f32
            / (libm::exp(value) + libm::exp(-value)) as f32
    }
    fn identity_function(value: f32) -> f32 {
        value
    }

    fn binary_step_function(value: f32) -> f32 {
        if value > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    fn relu_function(value: f32) -> f32 {
        if value > 0.0 {
            value
        } else {
            0.0
        }
    }

    fn softsign_function(value: f32) -> f32 {
        value / (1.0 + value.abs())
    }

    fn gaussian_function(value: f32) -> f32 {
        libm::exp(-value.powf(2.0) as f64) as f32
    }

    fn sinusiodial_function(value: f32) -> f32 {
        // a basic sinus function
        libm::sin(value as f64) as f32
    }

    fn bent_identity_function(value: f32) -> f32 {
        ((value.powf(2.0) + 1.0).sqrt() - 1.0) / 2.0 + value
    }

    fn selu_function(value: f32) -> f32 {
        let alpha: f32 = 1.67326;
        let lambda: f32 = 1.0507;
        if value < 0.0 {
            lambda * alpha * (libm::exp(value as f64) - 1.0) as f32
        } else {
            lambda * value
        }
    }

    pub fn get_output(squash: ActivationFunctions, value: f32) -> f32 {
        match squash {
            ActivationFunctions::Logistics => ActivationFunctions::logistic_function(value),
            ActivationFunctions::HyperbolicTangent => {
                ActivationFunctions::hyperbolic_tangent_function(value)
            }
            ActivationFunctions::Identity => ActivationFunctions::identity_function(value),
            ActivationFunctions::BinaryStep => ActivationFunctions::binary_step_function(value),
            ActivationFunctions::ReLU => ActivationFunctions::relu_function(value),
            ActivationFunctions::Softsign => ActivationFunctions::softsign_function(value),
            ActivationFunctions::Gaussian => ActivationFunctions::gaussian_function(value),
            ActivationFunctions::Sinusiodial => ActivationFunctions::sinusiodial_function(value),
            ActivationFunctions::BentIdentity => ActivationFunctions::bent_identity_function(value),
            ActivationFunctions::SELU => ActivationFunctions::selu_function(value),
            ActivationFunctions::None => value,
        }
    }
}

#[cfg(test)]
mod test_activation_functions {
    use super::*;

    #[test]
    fn test_logistic_function() {
        assert_eq!(ActivationFunctions::logistic_function(1.0).round(), 1.0);
        assert_eq!(ActivationFunctions::logistic_function(-0.5).round(), 0.0);
        assert_eq!(ActivationFunctions::logistic_function(0.0), 0.5);
    }

    #[test]
    fn test_hyperbolic_tangent_function() {
        assert_eq!(
            ActivationFunctions::hyperbolic_tangent_function(1.0).round(),
            1.0
        );
        assert_eq!(
            ActivationFunctions::hyperbolic_tangent_function(0.5).round(),
            0.0
        );
    }

    #[test]
    fn test_identity_function() {
        assert_eq!(ActivationFunctions::identity_function(1.0), 1.0);
    }

    #[test]
    fn test_binary_step_function() {
        assert_eq!(ActivationFunctions::binary_step_function(-1.0), 0.0);
        assert_eq!(ActivationFunctions::binary_step_function(1.0), 1.0);
    }

    #[test]
    fn test_relu_function() {
        assert_eq!(ActivationFunctions::relu_function(-1.0), 0.0);
        assert_eq!(ActivationFunctions::relu_function(1.0), 1.0);
    }

    #[test]
    fn test_softsign_function() {
        assert_eq!(ActivationFunctions::softsign_function(1.0), 0.5);
        assert_eq!(ActivationFunctions::softsign_function(-1.0), -0.5);
    }

    #[test]
    fn test_gaussian_function() {
        assert_eq!(ActivationFunctions::gaussian_function(1.0).round(), 0.0);
        assert_eq!(ActivationFunctions::gaussian_function(0.5).round(), 1.0);
    }

    #[test]
    fn test_sinusodial_function() {
        assert_eq!(ActivationFunctions::sinusiodial_function(1.0).round(), 1.0);
        assert_eq!(ActivationFunctions::sinusiodial_function(0.0), 0.0);
    }

    #[test]
    fn test_bent_identity_function() {
        assert_eq!(
            ActivationFunctions::bent_identity_function(1.0).round(),
            1.0
        );
        assert_eq!(ActivationFunctions::bent_identity_function(0.0), 0.0);
    }

    #[test]
    fn test_selu_function() {
        // alpha = 1.67326;
        // lambda = 1.0507;
        assert_eq!(ActivationFunctions::selu_function(-1.0).round(), -1.0);
        assert_eq!(ActivationFunctions::selu_function(1.0).round(), 1.0);
    }
}
