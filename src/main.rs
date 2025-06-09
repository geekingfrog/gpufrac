use anyhow::Context;
use clap::Parser;
use std::str::FromStr;
use wgpu::util::DeviceExt;

const U32_SIZE: u32 = std::mem::size_of::<u32>() as u32;

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
    clip_width: f32,
    _offset: f32,
    size_px: [f32; 2],
}

/// to store a bunch of useful data related to the dimension and size
/// of the textures and buffers
#[derive(Debug, Clone)]
struct Dimensions {
    /// the size in px of the image
    resolution: [u32; 2],
    /// the actual texture size. May differ from resolution for alignment
    texture_size: [u32; 2],
    /// used with copy_texture_to_buffer
    bytes_per_row: u32,
}

impl Dimensions {
    fn from_resolution(resolution: [u32; 2]) -> Self {
        let [width, height] = resolution;

        // To avoid
        // Bytes per row does not respect `COPY_BYTES_PER_ROW_ALIGNMENT`
        // the output buffer size must be properly padded. See
        // https://docs.rs/wgpu-types/latest/wgpu_types/constant.COPY_BYTES_PER_ROW_ALIGNMENT.html
        // Also see this PR for a project that had the same problem:
        // https://github.com/ggez/ggez/pull/1210/files

        let bytes_per_px = U32_SIZE;
        let unpadded_bytes_per_row = width * bytes_per_px;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + row_padding;

        let texture_width = width + (row_padding / bytes_per_px);

        Self {
            resolution,
            texture_size: [texture_width, height],
            bytes_per_row: padded_bytes_per_row,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct JuliaConstant {
    c: [f32; 2],
}

struct State {
    dimensions: Dimensions,
    device: wgpu::Device,
    queue: wgpu::Queue,
    render_pipeline: wgpu::RenderPipeline,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    output_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    // view: View,
    // view_buffer: wgpu::Buffer,
    view_bind_group: wgpu::BindGroup,
    // julia_constant: Option<[f32; 2]>,
    julia_bind_group: wgpu::BindGroup,
}

impl State {
    async fn new(run_opts: RunOpts) -> Self {
        let dimensions = Dimensions::from_resolution(run_opts.resolution.unwrap_or([1600, 900]));

        let view = View {
            clip_center: [-0.5, 0.0],
            clip_width: 2.5,
            _offset: 0.0,
            size_px: [dimensions.resolution[0] as _, dimensions.resolution[1] as _],
        };

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("request adapter");

        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .expect("request device");

        let texture_desc = wgpu::TextureDescriptor {
            label: Some("texture descriptor"),
            size: wgpu::Extent3d {
                width: dimensions.texture_size[0],
                height: dimensions.texture_size[1],
                depth_or_array_layers: 1,
            },
            view_formats: &[],
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let texture = device.create_texture(&texture_desc);
        let texture_view = texture.create_view(&Default::default());
        let output_buffer_size = dimensions.bytes_per_row * dimensions.texture_size[1] * U32_SIZE;

        let output_buffer_desc = wgpu::BufferDescriptor {
            size: output_buffer_size as _,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            //      ^ this tells wpgu that we want to read this buffer from the cpu
            label: None,
            mapped_at_creation: false,
        };
        let output_buffer = device.create_buffer(&output_buffer_desc);

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

        // let julia_constant = match run_opts.command {
        //     Command::Mandelbrot => None,
        //     Command::Julia { c } => Some(c),
        // };

        let cst = match &run_opts.command {
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

        let fragment_entry_point = match run_opts.command {
            Command::Mandelbrot => "fs_mandelbrot",
            Command::Julia { .. } => "fs_julia",
        };

        // let mut compiler = shaderc::Compiler::new().unwrap();

        let shader_src =
            std::fs::read_to_string("src/shader.wgsl").expect("cannot read shader source");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgsl shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&render_pipeline_layout), // where to put the bind groups for texture, camera and stuff
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fragment_entry_point),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_desc.format,
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
            cache: None,
        });

        State {
            dimensions,
            device,
            queue,
            texture,
            texture_view,
            output_buffer,
            render_pipeline,
            vertex_buffer,
            // view,
            // view_buffer,
            view_bind_group,
            // julia_constant,
            julia_bind_group,
        }
    }

    async fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
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
                    view: &self.texture_view,
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
            render_pass.set_bind_group(1, &self.julia_bind_group, &[]);
            // self.queue
            //     .write_buffer(&self.view_buffer, 0, bytemuck::cast_slice(&[self.view]));
            render_pass.draw(0..3, 0..1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    // bytes_per_row: Some(U32_SIZE * self.texture_size.0),
                    bytes_per_row: Some(self.dimensions.bytes_per_row),
                    rows_per_image: Some(self.dimensions.texture_size[1]),
                },
            },
            wgpu::Extent3d {
                width: self.dimensions.texture_size[0],
                height: self.dimensions.texture_size[1],
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // We need to scope the mapping variables so that we can
        // unmap the buffer
        {
            let buffer_slice = self.output_buffer.slice(..);

            // NOTE: We have to create the mapping THEN device.poll() before await
            // the future. Otherwise the application will freeze.
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            self.device
                .poll(wgpu::PollType::Wait)
                .expect("polling device");
            rx.receive().await.unwrap().unwrap();

            let data = buffer_slice.get_mapped_range();

            use image::{ImageBuffer, Rgba};
            let buffer = ImageBuffer::<Rgba<u8>, _>::from_raw(
                self.dimensions.texture_size[0],
                self.dimensions.texture_size[1],
                data,
            )
            .unwrap();
            buffer.save("image.png").unwrap();
        }
        self.output_buffer.unmap();

        Ok(())
    }
}

async fn run(run_opts: RunOpts) -> anyhow::Result<()> {
    let mut state = State::new(run_opts).await;
    state.render().await?;
    Ok(())
}

#[derive(clap::Parser)]
struct Opts {
    /// resolution of the image, for example 1024x748
    #[arg(short, long)]
    resolution: Option<String>,
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

struct RunOpts {
    command: Command,
    resolution: Option<[u32; 2]>,
}

impl TryFrom<Opts> for RunOpts {
    type Error = anyhow::Error;
    fn try_from(opts: Opts) -> Result<Self, Self::Error> {
        let resolution = match opts.resolution {
            Some(raw) => {
                let mut split = raw.split('x');
                let w = split.next();
                let h = split.next();
                match (w, h) {
                    (Some(w), Some(h)) => {
                        let w = u32::from_str_radix(w, 10)?;
                        let h = u32::from_str_radix(h, 10)?;
                        Some([w, h])
                    }
                    _ => anyhow::bail!("Invalid resolution, example: 1024x768"),
                }
            }
            None => None,
        };
        let command = opts.command.try_into()?;
        Ok(RunOpts {
            resolution,
            command,
        })
    }
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
    let run_opts = opts.try_into().context("parse command")?;
    pollster::block_on(run(run_opts)).expect("run!");
    Ok(())
}
