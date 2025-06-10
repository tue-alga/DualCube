use crate::render::{CameraFor, Objects};
use crate::{dual, to_color, ActionEvent, ActionEventStatus, CameraHandles, Configuration, InputResource, Perspective, SolutionResource};
use crate::{HexMeshStatus, PrincipalDirection};
use bevy::prelude::*;
use bevy_egui::egui::{emath::Numeric, text::LayoutJob, Align, Color32, FontId, Frame, Layout, Slider, TextFormat, TopBottomPanel, Ui, Window};
use bevy_egui::egui::{RichText, Rounding};

use enum_iterator::all;
use tico::tico;

pub fn setup(mut ui: bevy_egui::EguiContexts) {
    // Font
    let mut fonts = bevy_egui::egui::FontDefinitions::default();
    fonts
        .font_data
        .insert("font".to_owned(), bevy_egui::egui::FontData::from_static(include_bytes!("../assets/font.ttf")));
    fonts
        .families
        .entry(bevy_egui::egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "font".to_owned());
    fonts
        .families
        .entry(bevy_egui::egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "font".to_owned());
    ui.ctx_mut().set_fonts(fonts);

    // Theme
    ui.ctx_mut().set_visuals(bevy_egui::egui::Visuals::dark());

    ui.ctx_mut().style_mut(|style| {
        style.visuals.widgets.open.rounding = Rounding::same(0.);
        style.visuals.menu_rounding = Rounding::same(0.);
        style.visuals.window_rounding = Rounding::same(0.);
        style.visuals.widgets.noninteractive.rounding = Rounding::same(0.);
        style.visuals.widgets.hovered.rounding = Rounding::same(0.);
        style.visuals.widgets.active.rounding = Rounding::same(0.);
    });
}

fn header(
    ui: &mut Ui,
    ev_w: &mut EventWriter<ActionEvent>,
    mesh: &mut ResMut<InputResource>,
    solution: &Res<SolutionResource>,
    configuration: &mut ResMut<Configuration>,
    time: &Res<Time>,
) {
    ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
        Frame {
            outer_margin: bevy_egui::egui::epaint::Margin::symmetric(15., 0.),
            shadow: bevy_egui::egui::epaint::Shadow::NONE,
            ..default()
        }
        .show(ui, |ui| {
            bevy_egui::egui::menu::bar(ui, |ui| {
                menu_button(ui, "File", |ui| {
                    if sleek_button(ui, "Load") {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("triangulated geometry", &["obj", "stl", "save", "flag"])
                            .pick_file()
                        {
                            ev_w.send(ActionEvent::LoadFile(path));
                        }
                    }
                    ui.add_space(5.);
                    ui.separator();
                    ui.add_space(5.);
                    if sleek_button(ui, "Export SAVE") {
                        ev_w.send(ActionEvent::ExportState);
                    }
                    ui.add_space(5.);
                    if sleek_button(ui, "Export FLAG") {
                        ev_w.send(ActionEvent::ExportSolution);
                    }
                    ui.add_space(5.);
                    if sleek_button(ui, "Export NLR") {
                        ev_w.send(ActionEvent::ExportNLR);
                    }
                    ui.add_space(5.);
                    ui.separator();
                    ui.add_space(5.);
                    if sleek_button(ui, "Quit") {
                        std::process::exit(0);
                    }
                });

                ui.separator();

                menu_button(ui, "Info", |ui| {
                    if mesh.properties.source.is_empty() {
                        ui.label("No file loaded.");
                    } else {
                        ui.label(tico(
                            &mesh
                                .properties
                                .source
                                .to_string()
                                .chars()
                                .map(|ch| if ch == '\\' { '/' } else { ch })
                                .collect::<String>(),
                            None,
                        ));
                        ui.add_space(5.);
                        ui.label(format!(
                            "|Vm|: {}\n|Em|: {}\n|Fm|: {}",
                            mesh.mesh.nr_verts(),
                            mesh.mesh.nr_edges() / 2,
                            mesh.mesh.nr_faces()
                        ));
                    }

                    if let Some(polycube) = &solution.current_solution.polycube {
                        ui.add_space(5.);
                        ui.label(format!(
                            "|Vp|: {}\n|Ep|: {}\n|Fp|: {}",
                            polycube.structure.nr_verts(),
                            polycube.structure.nr_edges() / 2,
                            polycube.structure.nr_faces()
                        ));
                    }

                    ui.add_space(5.);
                    ui.separator();
                    ui.add_space(5.);

                    ui.label("CAMERA");
                    ui.add_space(2.);
                    ui.label("  Rotate: ctrl + right-mouse-drag");
                    ui.add_space(1.);
                    ui.label("  Pan: left-mouse-drag");
                    ui.add_space(1.);
                    ui.label("  Zoom: mouse-scroll");
                    ui.add_space(2.);
                    ui.separator();
                    ui.add_space(2.);
                    ui.label("MANUAL");
                    ui.add_space(2.);
                    ui.label("  Add loop: right-mouse-click");
                    ui.add_space(1.);
                    ui.label("  Delete loop: space + right-mouse-click");
                });

                ui.separator();

                menu_button(ui, "Rendering", |ui| {
                    ui.checkbox(&mut configuration.show_gizmos_mesh, "Wireframe");
                    ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                        ui.add_space(15.);
                        if configuration.show_gizmos_mesh {
                            ui.checkbox(&mut configuration.show_gizmos_mesh_granulated, "Wireframe granulated");
                        } else {
                            ui.checkbox(&mut false, "Wireframe granulated");
                        }
                    });

                    ui.add_space(5.);

                    ui.checkbox(&mut configuration.show_gizmos_loops[0], "x-loops");
                    ui.checkbox(&mut configuration.show_gizmos_loops[1], "y-loops");
                    ui.checkbox(&mut configuration.show_gizmos_loops[2], "z-loops");

                    ui.add_space(5.);

                    ui.checkbox(&mut configuration.show_gizmos_paths, "paths");
                    ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                        ui.add_space(15.);
                        if configuration.show_gizmos_paths {
                            ui.checkbox(&mut configuration.show_gizmos_flat_edges, "flat edges");
                        } else {
                            ui.checkbox(&mut false, "flat edges");
                        }
                    });
                    ui.add_space(5.);

                    ui.label("Background color (BUGGY: USE RIGHT or MIDDLE MOUSE BUTTON)");
                    // stupid bug: https://github.com/emilk/egui/issues/3718
                    ui.color_edit_button_srgb(&mut configuration.clear_color);

                    if sleek_button(ui, "Confirm") {
                        ev_w.send(ActionEvent::ResetCamera);
                        mesh.as_mut();
                    }
                });

                ui.separator();

                menu_button(ui, "Solution", |ui| {
                    ui.checkbox(&mut configuration.unit_cubes, "Contrain to unit grid");

                    if sleek_button(ui, "Recompute") {
                        ev_w.send(ActionEvent::Recompute);
                    }
                });

                ui.separator();

                menu_button(ui, "EXPERIMENTAL", |ui| {
                    ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                        if sleek_button(ui, "Smoothening") && !matches!(&configuration.smoothen_status, ActionEventStatus::Loading) {
                            ev_w.send(ActionEvent::Smoothen);
                        };

                        if matches!(configuration.smoothen_status, ActionEventStatus::Loading) {
                            ui.add_space(5.);
                            ui.label(text(&timer_animation(time)));
                        }
                    });

                    if let ActionEventStatus::Done(score) = &configuration.smoothen_status {
                        ui.add_space(5.);

                        ui.label(score);
                    }

                    ui.add_space(5.);

                    ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                        if sleek_button(ui, "Run hex-mesh pipeline") && !matches!(&configuration.hex_mesh_status, HexMeshStatus::Loading) {
                            ev_w.send(ActionEvent::ToHexmesh);
                        };
                        if matches!(configuration.hex_mesh_status, HexMeshStatus::Loading) {
                            ui.add_space(5.);
                            ui.label(text(&timer_animation(time)));
                        }
                    });

                    if let HexMeshStatus::Done(score) = &configuration.hex_mesh_status {
                        ui.add_space(5.);
                        ui.label("Hex-meshing results:");
                        ui.add_space(5.);
                        ui.label(format!(
                            "hd: {hd:.3}\nsJ: {sj:.3} ({sjmin:.3}-{sjmax:.3})\nirr: {irr:.3}",
                            hd = score.hausdorff,
                            sj = score.avg_jacob,
                            sjmin = score.min_jacob,
                            sjmax = score.max_jacob,
                            irr = score.irregular,
                        ));
                    }
                });
            });
        });
    });
}

fn footer(egui_ctx: &mut bevy_egui::EguiContexts, conf: &mut Configuration, solution: &SolutionResource, time: &Time) {
    TopBottomPanel::bottom("footer").show_separator_line(false).show(egui_ctx.ctx_mut(), |ui| {
        ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
            // Left side: Display FPS and loading animation
            ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                ui.add_space(15.);
                display_fps_and_loading(ui, conf.fps, time);
            });

            // Center: Display status of dual, embd, alignment, and orthogonality
            ui.vertical_centered(|ui| {
                let mut job = LayoutJob::default();
                append_status(&mut job, "dual", &solution.current_solution.dual);
                display_label(&mut job, " | ");
                append_status(&mut job, "primal", &solution.current_solution.layout);
                display_label(&mut job, " | ");
                display_label(&mut job, &format!("quality: {value:?}", value = solution.current_solution.get_quality()));
                ui.label(job);
            });

            // Right side: Display fixed label
            ui.with_layout(Layout::right_to_left(Align::TOP), |ui| {
                ui.add_space(15.);
                let mut job = LayoutJob::default();
                display_label(&mut job, "DualCube - maxim snoep");
                ui.label(job);
            });
        });

        conf.ui_is_hovered[1] = ui.ui_contains_pointer();
    });
}

fn display_fps_and_loading(ui: &mut Ui, fps: f64, time: &Time) {
    let mut job = LayoutJob::default();
    job.append(&format!("{fps:.0}"), 0.0, text_format(9.0, Color32::WHITE));
    ui.label(job);
}

fn append_status<T>(job: &mut LayoutJob, label: &str, result: &Result<T, dual::PropertyViolationError>) {
    job.append(&format!("{label}: "), 0.0, text_format(9.0, Color32::WHITE));
    match result {
        Ok(_) => job.append("Ok", 0.0, text_format(9.0, Color32::GREEN)),
        Err(err) => job.append(&format!("{err:?}"), 0.0, text_format(9.0, Color32::RED)),
    }
}

fn display_label(job: &mut LayoutJob, label: &str) {
    job.append(label, 0.0, text_format(9.0, Color32::WHITE));
}

pub fn update(
    mut egui_ctx: bevy_egui::EguiContexts,
    mut ev_w: EventWriter<ActionEvent>,
    mut conf: ResMut<Configuration>,
    mut mesh: ResMut<InputResource>,
    solution: Res<SolutionResource>,
    time: Res<Time>,
    image_handle: Res<CameraHandles>,
) {
    TopBottomPanel::top("panel").show_separator_line(false).show(egui_ctx.ctx_mut(), |ui| {
        ui.add_space(10.);

        ui.horizontal(|ui| {
            ui.with_layout(Layout::top_down(Align::TOP), |ui| {
                // FIRST ROW
                header(ui, &mut ev_w, &mut mesh, &solution, &mut conf, &time);

                ui.add_space(5.);

                ui.separator();

                ui.add_space(5.);

                // NEXT ROW
                ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                    ui.add_space(15.);
                });

                ui.add_space(5.);

                // THIRD ROW
                ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                    ui.add_space(15.);

                    bevy_egui::egui::menu::bar(ui, |ui| {
                        if if conf.automatic {
                            sleek_button(ui, "AUTO")
                        } else {
                            sleek_button_unfocused(ui, "AUTO")
                        } {
                            if conf.automatic {
                                conf.automatic = false;
                            } else {
                                conf.automatic = true;
                                conf.interactive = false;
                            }
                            conf.should_continue = false;
                        };

                        let rt: RichText = RichText::new("|").color(Color32::GRAY);
                        ui.label(rt);

                        if if conf.interactive {
                            sleek_button(ui, "MANUAL")
                        } else {
                            sleek_button_unfocused(ui, "MANUAL")
                        } {
                            if conf.interactive {
                                conf.interactive = false;
                            } else {
                                conf.interactive = true;
                                conf.automatic = false;
                            }
                            conf.should_continue = false;
                        };

                        ui.add_space(15.);

                        if conf.automatic {
                            if sleek_button(ui, "initialize") {
                                ev_w.send(ActionEvent::Initialize);
                            }

                            if sleek_button(ui, "mutate") {
                                ev_w.send(ActionEvent::Mutate);
                            }
                        }

                        if conf.interactive {
                            for direction in [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z] {
                                radio(
                                    ui,
                                    &mut conf.direction,
                                    direction,
                                    Color32::from_rgb(
                                        (to_color(direction, Perspective::Dual, None)[0] * 255.) as u8,
                                        (to_color(direction, Perspective::Dual, None)[1] * 255.) as u8,
                                        (to_color(direction, Perspective::Dual, None)[2] * 255.) as u8,
                                    ),
                                );
                            }

                            // add slider for alpha (or 1-beta)
                            slider(ui, "alpha", &mut conf.alpha, 0.0..=1.0);

                            if let Some(edgepair) = conf.selected {
                                if let Some(Some(sol)) = solution.next[conf.direction as usize].get(&edgepair) {
                                    ui.label("DUAL[");
                                    if sol.dual.is_ok() {
                                        ui.label(colored_text("Ok", Color32::GREEN));
                                    } else {
                                        ui.label(colored_text(&format!("{:?}", sol.dual.as_ref().err()), Color32::RED));
                                    }
                                    ui.label("]");

                                    ui.label("EMBD[");
                                    if sol.layout.is_ok() {
                                        ui.label(colored_text("Ok", Color32::GREEN));
                                    } else {
                                        ui.label(colored_text(&format!("{:?}", sol.layout.as_ref().err()), Color32::RED));
                                    }

                                    ui.label("]");

                                    if let Some(alignment) = sol.alignment {
                                        ui.label("ALIGN[");
                                        ui.label(format!("{alignment:.3}"));
                                        ui.label("]");
                                    }

                                    if let Some(orthogonality) = sol.orthogonality {
                                        ui.label("ORTH[");
                                        ui.label(format!("{orthogonality:.3}"));
                                        ui.label("]");
                                    }
                                }
                            }
                        }
                    });
                });

                ui.add_space(5.);
            });
        });

        conf.ui_is_hovered[0] = ui.ui_contains_pointer();
    });

    footer(&mut egui_ctx, &mut conf, &solution, &time);

    bevy_egui::egui::CentralPanel::default()
        .frame(Frame {
            outer_margin: bevy_egui::egui::epaint::Margin::same(10.),
            stroke: bevy_egui::egui::epaint::Stroke {
                width: 10.0,
                color: Color32::from_rgb(27, 27, 27),
            },
            shadow: bevy_egui::egui::epaint::Shadow::NONE,
            ..default()
        })
        .show(egui_ctx.ctx_mut(), |ui| {
            Frame {
                outer_margin: bevy_egui::egui::epaint::Margin::same(1.),
                stroke: bevy_egui::egui::epaint::Stroke {
                    width: 1.0,
                    color: Color32::from_rgb(50, 50, 50),
                },
                shadow: bevy_egui::egui::epaint::Shadow::NONE,
                ..default()
            }
            .show(ui, |ui| {
                ui.allocate_space(ui.available_size());
            });
        });

    for i in 0..3 {
        let egui_handle = egui_ctx.add_image(image_handle.map.get(&CameraFor(conf.window_shows_object[i])).unwrap().clone());
        Window::new(format!("window {i}"))
            .frame(Frame {
                outer_margin: bevy_egui::egui::epaint::Margin::same(2.),
                stroke: bevy_egui::egui::epaint::Stroke {
                    width: 1.0,
                    color: Color32::from_rgb(50, 50, 50),
                },
                shadow: bevy_egui::egui::epaint::Shadow::NONE,
                fill: Color32::from_rgb(27, 27, 27),
                ..default()
            })
            .max_size([conf.window_has_size[i], conf.window_has_size[i]])
            .title_bar(false)
            .id(format!("window {i}").into())
            .resizable([false, false])
            .show(egui_ctx.ctx_mut(), |ui| {
                Frame {
                    outer_margin: bevy_egui::egui::epaint::Margin::symmetric(15., 0.),
                    shadow: bevy_egui::egui::epaint::Shadow::NONE,
                    ..default()
                }
                .show(ui, |ui| {
                    bevy_egui::egui::menu::bar(ui, |ui| {
                        menu_button(ui, &String::from(conf.window_shows_object[i]), |ui| {
                            for object in all::<Objects>() {
                                if ui.button(String::from(object)).clicked() {
                                    conf.window_shows_object[i] = object;
                                }
                            }
                        });

                        if sleek_button(ui, "+") {
                            if conf.window_has_size[i] > 0. {
                                conf.window_has_size[i] += 128.;
                                if conf.window_has_size[i] > 640. {
                                    conf.window_has_size[i] = 640.;
                                }
                            } else {
                                conf.window_has_size[i] = 256.;
                            }
                        }
                        if sleek_button(ui, "-") {
                            conf.window_has_size[i] -= 128.;
                            if conf.window_has_size[i] < 256. {
                                conf.window_has_size[i] = 0.;
                            }
                        }
                    });
                });

                ui.add(bevy_egui::egui::widgets::Image::new(bevy_egui::egui::load::SizedTexture::new(
                    egui_handle,
                    [conf.window_has_size[i], conf.window_has_size[i]],
                )));

                conf.ui_is_hovered[31 - i] = ui.ui_contains_pointer() || ui.ctx().is_being_dragged(format!("window {i}").into());
            });
    }
}

fn slider<T: Numeric>(ui: &mut Ui, label: &str, value: &mut T, range: std::ops::RangeInclusive<T>) {
    ui.add(Slider::new(value, range).text(text(label)));
}

fn stepper(ui: &mut Ui, label: &str, value: &mut u32, min: u32, max: u32) -> bool {
    ui.horizontal(|ui| {
        if ui.button("<<").clicked() {
            let new_value = *value - 1;
            if new_value >= min && new_value <= max {
                *value = new_value;
            } else {
                *value = max;
            };
            return true;
        }
        ui.label(format!("{label}: {value} [{min}-{max}]"));
        if ui.button(">>").clicked() {
            let new_value = *value + 1;
            if new_value <= max && new_value >= min {
                *value = new_value;
            } else {
                *value = min;
            };
            return true;
        }
        false
    })
    .inner
}

fn radio<T: PartialEq<T> + std::fmt::Display>(ui: &mut Ui, item: &mut T, value: T, color: Color32) -> bool {
    if ui.radio(*item == value, colored_text(&format!("{value}"), color)).clicked() {
        *item = value;
        true
    } else {
        false
    }
}

pub fn text(string: &str) -> LayoutJob {
    colored_text(string, Color32::WHITE)
}

pub fn colored_text(string: &str, color: Color32) -> LayoutJob {
    let mut job = LayoutJob::default();
    job.append(string, 0.0, text_format(13.0, color));
    job
}

pub fn text_format(size: f32, color: Color32) -> TextFormat {
    TextFormat {
        font_id: FontId {
            size,
            family: bevy_egui::egui::FontFamily::Monospace,
        },
        color,
        ..Default::default()
    }
}

pub fn menu_button(ui: &mut Ui, label: &str, f: impl FnOnce(&mut Ui)) {
    bevy_egui::egui::menu::menu_button(ui, RichText::new(label).color(Color32::WHITE), f);
}

pub fn menu_button_unfocused(ui: &mut Ui, label: &str, f: impl FnOnce(&mut Ui)) {
    bevy_egui::egui::menu::menu_button(ui, RichText::new(label).color(Color32::GRAY), f);
}

pub fn sleek_button(ui: &mut Ui, label: &str) -> bool {
    bevy_egui::egui::menu::menu_button(ui, RichText::new(label).color(Color32::WHITE), |ui| {
        ui.close_menu();
    })
    .response
    .clicked()
}

pub fn sleek_button_unfocused(ui: &mut Ui, label: &str) -> bool {
    bevy_egui::egui::menu::menu_button(ui, RichText::new(label).color(Color32::GRAY), |ui| {
        ui.close_menu();
    })
    .response
    .clicked()
}

pub fn timer_animation(time: &Time) -> String {
    let frequency = 5.;
    let animation = ["◐", "◓", "◑", "◒"];
    let index = (time.elapsed_seconds() * frequency) as usize % animation.len();
    animation[index].to_string()
}
