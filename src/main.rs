use anyhow::Context;
use clap::Parser;
use std::str::FromStr;
use wgpu::util::DeviceExt;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

impl From<[f32; 2]> for Vertex {
    fn from(value: [f32; 2]) -> Self {
        Vertex {
            pos: [value[0], value[1], 0.0],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct View {
    clip_center: [f32; 2],
    size_px: [f32; 2],
    zoom: f32,
    _offset: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct JuliaConstant {
    c: [f32; 2],
}

struct State {
    device: wgpu::Device,
    config: wgpu::SurfaceConfiguration,
    queue: wgpu::Queue,
    render_pipeline: wgpu::RenderPipeline,
    surface: wgpu::Surface,
    vertex_buffer: wgpu::Buffer,
    view: View,
    view_buffer: wgpu::Buffer,
    view_bind_group: wgpu::BindGroup,
    julia_constant: Option<[f32; 2]>,
    julia_bind_group: wgpu::BindGroup,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    window: Window,
}

impl State {
    async fn new(cmd: Command, window: Window) -> Self {
        let window_size = window.inner_size();
        let view = View {
            clip_center: [-0.3, 0.0],
            size_px: [window_size.width as _, window_size.height as _],
            zoom: 0.6,
            _offset: 0.0,
        };

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("request adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("request device");

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: window_size.width,
            height: window_size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // It seems naga doesn't accept hardcoding the vertices in the shader directly
        // and using the index to get the position, so instead pass the vertices
        // through a vertex buffer
        let vertices = &[
            Vertex::from([-1.0, -1.0]),
            Vertex::from([3.0, -1.0]),
            Vertex::from([-1.0, 3.0]),
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let view_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view uniform buffer"),
            contents: bytemuck::cast_slice(&[view]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });

        let view_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("window size bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let view_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view bind group layout"),
            layout: &view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view_buffer.as_entire_binding(),
            }],
        });

        let julia_constant = match cmd {
            Command::Mandelbrot => None,
            Command::Julia { c } => Some(c),
        };

        let cst = match &cmd {
            Command::Mandelbrot => JuliaConstant { c: [0.0, 0.0] },
            Command::Julia { c } => JuliaConstant { c: c.clone() },
        };

        let julia_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("julia uniform buffer"),
            contents: bytemuck::cast_slice(&[cst]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });

        let julia_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("julia bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let julia_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("julia bind group"),
            layout: &julia_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: julia_buffer.as_entire_binding(),
            }],
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render pipeline layout"),
                bind_group_layouts: &[&view_bind_group_layout, &julia_bind_group_layout],
                push_constant_ranges: &[],
            });

        let fragment_entry_point = match cmd {
            Command::Mandelbrot => "fs_mandelbrot",
            Command::Julia { .. } => "fs_julia",
        };

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&render_pipeline_layout), // where to put the bind groups for texture, camera and stuff
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: fragment_entry_point,
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        State {
            device,
            config,
            queue,
            render_pipeline,
            vertex_buffer,
            view,
            view_buffer,
            view_bind_group,
            julia_constant,
            julia_bind_group,
            surface,
            window,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.view.size_px = new_size.into();
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render encoder"),
            });

        {
            let clear_color = wgpu::Color {
                r: 0.0,
                g: 0.6,
                b: 0.4,
                a: 1.0,
            };
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.view_bind_group, &[]);
            if let Some(_c) = self.julia_constant {
                render_pass.set_bind_group(1, &self.julia_bind_group, &[]);
            };
            // self.queue
            //     .write_buffer(&self.view_buffer, 0, bytemuck::cast_slice(&[self.view]));
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

async fn run(cmd: Command) -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new().build(&event_loop)?;
    let mut state = State::new(cmd, window).await;
    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop
        .run(move |loop_event, elwt| match loop_event {
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                // let start = std::time::Instant::now();
                state.render().expect("render!");
                // println!("render took {:?}", start.elapsed());
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => elwt.exit(),
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => state.resize(size),
            _ => (),
        })
        .context("event loop run")?;
    Ok(())
}

#[derive(clap::Parser)]
struct Opts {
    #[command(subcommand)]
    command: OptsCommand,
}

#[derive(Clone, clap::Subcommand)]
enum OptsCommand {
    Mandelbrot,
    Julia {
        /// constant to use, as 2 comma separated floats: -0.5251993,-0.5251993
        c: String,
    },
}

enum Command {
    Mandelbrot,
    Julia {
        /// julia constant to use
        c: [f32; 2],
    },
}

impl TryFrom<OptsCommand> for Command {
    type Error = anyhow::Error;
    fn try_from(value: OptsCommand) -> Result<Self, Self::Error> {
        match value {
            OptsCommand::Mandelbrot => Ok(Command::Mandelbrot),
            OptsCommand::Julia { c } => {
                let mut s = c.split(',');
                let cr = s.next();
                let ci = s.next();
                match (cr, ci) {
                    (Some(cr), Some(ci)) => {
                        let cr = f32::from_str(cr)?;
                        let ci = f32::from_str(ci)?;
                        Ok(Command::Julia { c: [cr, ci] })
                    }
                    _ => Err(anyhow::anyhow!("Invalid constant, example: -0.52,0.21")),
                }
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    let opts = Opts::parse();
    let cmd: Command = opts.command.try_into().context("Parse command")?;
    pollster::block_on(run(cmd)).expect("run!");
    Ok(())
}
