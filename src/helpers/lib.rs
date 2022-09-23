use rand::Rng;

// random number between 0 and 1 centered on 0.5
pub fn gaussian_rand() -> f32 {
    let range = 6;
    let mut rng = rand::thread_rng();
    let mut random: f32 = 0.0;

    for _ in 0..range {
        random += rng.gen::<f32>();
    }

    return random / range as f32;
}

// takes the range of guassianRand and makes it [-1, 1]
pub fn std0() -> f32 {
    (gaussian_rand() - 0.5) * 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // This test should give a random number between 0 and 1
    fn test_guessian_rand() {
        let num = gaussian_rand();
        assert!(num >= 0.0 && num <= 1.0);
    }

    #[test]
    // return a number between 0 and 1 to -1 to 1
    fn test_std0() {
        let num = std0();
        assert!(num >= -1.0 && num <= 1.0)
    }
}
