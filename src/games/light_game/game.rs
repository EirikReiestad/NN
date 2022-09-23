use ggez;
use ggez::event;
use ggez::event::{KeyCode, KeyMods};
use ggez::graphics;
use ggez::{Context, GameResult};

use super::super::super::neural_network::environment;

pub struct Board {
    pub board: Vec<Vec<u8>>,
    pub last_chosen: (usize, usize),
}

impl Board {
    pub fn new(size: usize) -> Self {
        let mut board = vec![];
        let sqrt_size = f32::sqrt(size as f32).floor();

        let mut tmp = vec![];
        let mut count = 1;
        for _ in 0..size as u32 {
            tmp.push(0);
            if count as f32 % sqrt_size == 0.0 {
                board.push(tmp);
                tmp = vec![];
            }
            count += 1;
        }
        Board {
            board: board,
            last_chosen: (0, 0),
        }
    }

    pub fn check_finish(&mut self) -> (bool, u32) {
        let mut total = 0;
        let mut res = true;
        self.board.iter().for_each(|row| {
            row.iter().for_each(|col| {
                if col == &0_u8 {
                    res = false
                } else {
                    total += 1;
                };
            })
        });
        (res, total)
    }

    pub fn update_tile(&mut self, index: usize) -> bool {
        let len = self.board.len();
        let row = index / len;
        let col = index % len;
        match self.board[row][col] {
            0 => self.board[row][col] = 1,
            1 => self.board[row][col] = 0,
            _ => (),
        }
        match self.board[row][col] {
            1 => true,
            0 => false,
            _ => false,
        }
    }
}

struct MainState {
    env: environment::Environment,
    boards: Vec<Board>,
    round: u32,
    input_size: usize,
    draw_no: u32,
    best: u32,
    all_time_best: u32,
    last_best: u32,
}

// env is the environment
// board is a list of boards, the index is associated with the species at same index in env
// round tells how many round or moves that has been produced in this generation

impl MainState {
    pub fn new(
        ctx: &mut Context,
        env: environment::Environment,
        input_size: usize,
        draw_no: u32,
    ) -> Self {
        graphics::set_window_title(ctx, "LIGHT GAME");

        let mut boards = vec![];

        for _ in 0..env.species.len() {
            boards.push(Board::new(input_size))
        }

        let mut draw_no = draw_no;
        if draw_no > env.species.len() as u32 {
            draw_no = env.species.len() as u32
        }

        MainState {
            env: env,
            boards: boards,
            input_size: input_size,
            round: 0,
            draw_no: draw_no,
            best: 0,
            all_time_best: 0,
            last_best: 0,
        }
    }
}

impl event::EventHandler for MainState {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        if self.round as f32
            > self.boards[0].board[0].len() as f32 * self.boards[0].board.len() as f32 * 1.5
        {
            let mut new_boards = vec![];
            for _ in 0..self.env.species.len() {
                new_boards.push(Board::new(self.input_size));
            }
            self.boards = new_boards;

            if self.env.generation - self.last_best > 1000 {
                self.env.next_generation(true);
                self.last_best = self.env.generation;
                self.best = 0;
            } else {
                self.env.next_generation(false);
            }
            self.round = 0;
        }
        if self.best > self.all_time_best {
            self.all_time_best = self.best;
        }
        for i in 0..self.env.species.len() {
            let (res, total) = self.boards[i].check_finish();
            if total > self.best {
                self.best = total;
                self.last_best = self.env.generation;
            }
            if res {
                // std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
            if self.boards[i].last_chosen.1 > 4 {
                self.env.species[i].reward(-1.0);
                continue;
            }

            let mut input_values: Vec<f32> = vec![];
            self.boards[i]
                .board
                .iter()
                .for_each(|row| row.iter().for_each(|col| input_values.push(*col as f32)));

            // Penalty or reward

            let index_out = self.env.species[i].update(input_values);
            if self.boards[i].update_tile(index_out) {
                // if it turns on a tile, give reward
                self.env.species[i].reward(1.0);
            } else {
                self.env.species[i].reward(-1.0);
            }
            // gives them a reward of -1.0 for every round
            // self.env.species[i].reward(-1.0);

            // if the same square is chosen, give them a penalty
            if index_out == self.boards[i].last_chosen.0 {
                self.boards[i].last_chosen.1 += 1;
                // self.env.species[i].reward(-1.0);
            }
            self.boards[i].last_chosen.1 = index_out;
        }
        self.round += 1;
        Ok(())
    }
    fn key_down_event(
        &mut self,
        _ctx: &mut Context,
        keycode: KeyCode,
        _keymods: KeyMods,
        _repeat: bool,
    ) {
        match keycode {
            KeyCode::Up => {
                self.boards[0].update_tile(0);
            }
            KeyCode::Down => {
                self.boards[1].update_tile(0);
            }
            KeyCode::Right => {
                self.boards[2].update_tile(0);
            }
            KeyCode::Left => {
                self.boards[3].update_tile(0);
            }
            _ => {}
        }
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let (screen_w, screen_h) = graphics::drawable_size(ctx);
        let canvas_size = f32::sqrt(self.boards[0..self.draw_no as usize].len() as f32);
        let canvas_w = screen_w / canvas_size;
        let canvas_h = screen_h / canvas_size;
        let square_w = canvas_w / self.boards[0].board[0].len() as f32;
        let square_h = canvas_h / self.boards[0].board.len() as f32;
        graphics::clear(ctx, graphics::WHITE);

        let mut count = 0;

        let mut canvas_x = 0;
        let mut canvas_y = 0;
        for canvas in &self.boards[..self.draw_no as usize] {
            let new_canvas = graphics::Rect::new(
                canvas_x as f32 * canvas_w,
                canvas_y as f32 * canvas_h,
                canvas_w,
                canvas_h,
            );
            let canvas_boarder_mesh = graphics::Mesh::new_rectangle(
                ctx,
                graphics::DrawMode::stroke(4.0),
                new_canvas,
                graphics::Color::new(1.0, 0.0, 0.0, 1.0),
            )?;

            let draw_param_canvas = graphics::DrawParam::default();

            graphics::draw(ctx, &canvas_boarder_mesh, draw_param_canvas)?;

            let mut board_x = 0;
            let mut board_y = 0;
            for row in &canvas.board {
                for col in row {
                    let mut color = graphics::WHITE;
                    if col == &1 {
                        color = graphics::BLACK;
                    };
                    let new_tile = graphics::Rect::new(
                        board_x as f32 * square_w + canvas_w * canvas_x as f32,
                        board_y as f32 * square_h + canvas_h * canvas_y as f32,
                        square_w,
                        square_h,
                    );
                    let tile_boarder_mesh = graphics::Mesh::new_rectangle(
                        ctx,
                        graphics::DrawMode::stroke(2.0),
                        new_tile,
                        graphics::Color::new(0.0, 0.0, 0.0, 1.0),
                    )?;

                    let tile_mesh = graphics::Mesh::new_rectangle(
                        ctx,
                        graphics::DrawMode::fill(),
                        new_tile,
                        color,
                    )?;

                    let draw_param_tile = graphics::DrawParam::default();

                    graphics::draw(ctx, &tile_boarder_mesh, draw_param_tile)?;
                    graphics::draw(ctx, &tile_mesh, draw_param_tile)?;
                    board_x += 1;
                }
                board_x = 0;
                board_y += 1;
            }
            canvas_x += 1;
            count += 1;
            if count % canvas_size.floor() as usize == 0 {
                canvas_y += 1;
                canvas_x = 0;
                count = 0;
            }
        }

        // write the generation
        let generation_text =
            graphics::Text::new(String::from("Generation: ") + &self.env.generation.to_string());
        let param_generation_text =
            graphics::DrawParam::default().color(graphics::Color::new(1.0, 0.0, 0.0, 1.0));
        graphics::draw(ctx, &generation_text, param_generation_text)?;

        // best score
        let best_score_text =
            graphics::Text::new(String::from("Best score: ") + &self.best.to_string());
        let param_best_score_text = graphics::DrawParam::default()
            .color(graphics::Color::new(0.0, 1.0, 0.0, 1.0))
            .dest([150.0, 0.0]);
        graphics::draw(ctx, &best_score_text, param_best_score_text)?;
        let all_time_best_score_text = graphics::Text::new(
            String::from("All time best score: ") + &self.all_time_best.to_string(),
        );
        let param_all_time_best_score_text = graphics::DrawParam::default()
            .color(graphics::Color::new(0.0, 0.0, 1.0, 1.0))
            .dest([300.0, 0.0]);
        graphics::draw(
            ctx,
            &all_time_best_score_text,
            param_all_time_best_score_text,
        )?;

        graphics::present(ctx)?;
        Ok(())
    }
}

pub fn run(num_species: u32, num_input: u32, num_output: u32, draw_no: u32) -> GameResult {
    let env = environment::Environment::new(num_species, num_input, num_output);

    let game = ggez::ContextBuilder::new("light game", "light game");
    let (ctx, event_loop) = &mut game.build()?;

    let mut state = MainState::new(ctx, env, num_output as usize, draw_no);
    event::run(ctx, event_loop, &mut state)?;
    Ok(())
}
