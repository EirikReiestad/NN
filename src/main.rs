mod games;
mod helpers;
mod neural_network;

fn main() {
    match games::light_game::game::run(200, 9, 9, 1) {
        Ok(_) => (),
        Err(e) => panic!("{:?}", e),
    }
}