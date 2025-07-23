use clap::Parser;
use ggez::{
    Context, GameResult,
    conf::{WindowMode, WindowSetup},
    event, glam,
    graphics::{self, Color, DrawMode, Mesh, MeshBuilder},
};
use nalgebra::Point2;
use quadruped::legs::planar::KinematicState;

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(value_parser = |s:&str| serde_json::from_str::<KinematicState<f64>>(s))]
    kinematic_state: Option<KinematicState<f64>>,
}

struct MainState {
    mesh: Mesh,
}

fn convert_pt(p: Point2<f64>) -> glam::Vec2 {
    glam::Vec2::new(300.0 + 20.0 * -p.x as f32, 200.0 + 20.0 * -p.y as f32)
}

fn kinematic_state_mesh(ctx: &mut Context, ks: &KinematicState<f64>) -> GameResult<Mesh> {
    let line_color = (128, 128, 128).into();
    let fill_color = (64, 64, 64).into();
    let point_radius = 8.0;
    Ok(Mesh::from_data(
        ctx,
        MeshBuilder::new()
            .polyline(
                DrawMode::stroke(2.0),
                &[
                    convert_pt(ks.point_a),
                    convert_pt(ks.point_c),
                    convert_pt(ks.point_e),
                ],
                line_color,
            )?
            .polyline(
                DrawMode::stroke(2.0),
                &[convert_pt(ks.point_b), convert_pt(ks.point_d)],
                line_color,
            )?
            .polygon(
                DrawMode::fill(),
                &[
                    convert_pt(ks.point_d),
                    convert_pt(ks.point_e),
                    convert_pt(ks.point_f),
                ],
                fill_color,
            )?
            .polygon(
                DrawMode::stroke(2.0),
                &[
                    convert_pt(ks.point_d),
                    convert_pt(ks.point_e),
                    convert_pt(ks.point_f),
                ],
                line_color,
            )?
            .circle(
                DrawMode::stroke(2.0),
                convert_pt(ks.point_a),
                point_radius,
                1.0,
                (0xff, 0x00, 0x00).into(),
            )?
            .circle(
                DrawMode::stroke(2.0),
                convert_pt(ks.point_a),
                point_radius * 1.5,
                1.0,
                (0xff, 0x00, 0x00).into(),
            )?
            .circle(
                DrawMode::stroke(2.0),
                convert_pt(ks.point_b),
                point_radius,
                1.0,
                (0x00, 0xff, 0x00).into(),
            )?
            .circle(
                DrawMode::stroke(3.0),
                convert_pt(ks.point_b),
                point_radius * 1.5,
                1.0,
                (0x00, 0xff, 0x00).into(),
            )?
            .circle(
                DrawMode::stroke(3.0),
                convert_pt(ks.point_c),
                point_radius,
                1.0,
                (0x00, 0x00, 0xff).into(),
            )?
            .circle(
                DrawMode::stroke(3.0),
                convert_pt(ks.point_d),
                point_radius,
                1.0,
                (0xff, 0x66, 0x00).into(),
            )?
            .circle(
                DrawMode::stroke(3.0),
                convert_pt(ks.point_e),
                point_radius,
                1.0,
                (0x66, 0x00, 0xff).into(),
            )?
            .circle(
                DrawMode::stroke(3.0),
                convert_pt(ks.point_f),
                point_radius,
                1.0,
                (0xff, 0xff, 0xff).into(),
            )?
            .build(),
    ))
}

/// Kinematic state taken from the sketch in leg-schematic.svg
fn example_ks() -> KinematicState<f64> {
    KinematicState {
        point_a: Point2::new(0.0, 0.0),
        point_b: Point2::new(-6.0, -7.0),
        point_c: Point2::new(-9.0, 4.0),
        point_d: Point2::new(-16.0, -11.0),
        point_e: Point2::new(-26.0, -5.0),
        point_f: Point2::new(-4.0, -23.0),
    }
}

impl MainState {
    fn new(ctx: &mut Context, ks: &KinematicState<f64>) -> GameResult<MainState> {
        let mesh = kinematic_state_mesh(ctx, ks)?;
        Ok(MainState { mesh })
    }
}

impl event::EventHandler<ggez::GameError> for MainState {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, Color::from((0xa, 0xa, 0xa)));

        canvas.draw(&self.mesh, glam::Vec2::new(0.0, 0.0));

        canvas.finish(ctx)?;

        Ok(())
    }
}

pub fn main() -> GameResult {
    let args = Args::parse();

    let (mut ctx, event_loop) = ggez::ContextBuilder::new("Leg 2D visualisation", "cube")
        .window_setup(WindowSetup::default().title("2D leg visualisation"))
        .window_mode(WindowMode::default().resizable(true))
        .build()?;
    let state = MainState::new(&mut ctx, &args.kinematic_state.unwrap_or_else(example_ks))?;
    event::run(ctx, event_loop, state);
}
