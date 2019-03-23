use arrayvec::ArrayVec;
use core::ops::Deref;
#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;
use gfx_hal::buffer::IndexBufferView;
use gfx_hal::image::SubresourceLayers;
use gfx_hal::memory::Properties;
use gfx_hal::memory::Requirements;
use gfx_hal::pass::Subpass;
use gfx_hal::pso::AttributeDesc;
use gfx_hal::pso::BakedStates;
use gfx_hal::pso::BasePipeline;
use gfx_hal::pso::BlendDesc;
use gfx_hal::pso::BlendOp;
use gfx_hal::pso::BlendState;
use gfx_hal::pso::ColorBlendDesc;
use gfx_hal::pso::ColorMask;
use gfx_hal::pso::DepthStencilDesc;
use gfx_hal::pso::DepthTest;
use gfx_hal::pso::DescriptorSetLayoutBinding;
use gfx_hal::pso::ElemOffset;
use gfx_hal::pso::Element;
use gfx_hal::pso::EntryPoint;
use gfx_hal::pso::Face;
use gfx_hal::pso::Factor;
use gfx_hal::pso::FrontFace;
use gfx_hal::pso::GraphicsPipelineDesc;
use gfx_hal::pso::GraphicsShaderSet;
use gfx_hal::pso::InputAssemblerDesc;
use gfx_hal::pso::LogicOp;
use gfx_hal::pso::PipelineCreationFlags;
use gfx_hal::pso::PolygonMode;
use gfx_hal::pso::PrimitiveRestart::U16;
use gfx_hal::pso::Rasterizer;
use gfx_hal::pso::ShaderStageFlags;
use gfx_hal::pso::Specialization;
use gfx_hal::pso::StencilTest;
use gfx_hal::pso::VertexBufferDesc;
use gfx_hal::pso::Viewport;
use gfx_hal::Capability;
use gfx_hal::CommandQueue;
use gfx_hal::IndexType;
use gfx_hal::MemoryType;
use gfx_hal::MemoryTypeId;
use gfx_hal::Primitive;
use gfx_hal::Supports;
use gfx_hal::Transfer;
use gfx_hal::{
    adapter::{Adapter, PhysicalDevice},
    buffer::Usage as BufferUsage,
    command::{ClearColor, ClearValue, CommandBuffer, MultiShot, Primary},
    device::Device,
    format::{Aspects, ChannelType, Format, Swizzle},
    image::{Extent, Layout, SubresourceRange, Usage, ViewKind},
    pass::{Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, SubpassDesc},
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::{PipelineStage, Rect},
    queue::{family::QueueGroup, Submission},
    window::{Backbuffer, Extent2D, FrameSync, PresentMode, Swapchain, SwapchainConfig},
    Backend, DescriptorPool, Gpu, Graphics, Instance, QueueFamily, Surface,
};
use nalgebra_glm as glm;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use gfx_hal::memory::Pod;
use gfx_hal::pso::ElemStride;
use gfx_hal::pso::PrimitiveRestart;
use nalgebra_glm::sin;
use shaderc;
use std::collections::HashSet;
use std::marker::PhantomData;
use std::mem::size_of;
use std::mem::size_of_val;
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::time::Instant;
use winit::dpi::LogicalSize;
use winit::CreationError;
use winit::DeviceEvent;
use winit::ElementState;
use winit::Event;
use winit::EventsLoop;
use winit::KeyboardInput;
use winit::VirtualKeyCode;
use winit::Window;
use winit::WindowBuilder;
use winit::WindowEvent;

static VERTEX_SOURCE: &'static str = include_str!("vert.glsl");
static FRAGMENT_SOURCE: &'static str = include_str!("fragment.glsl");
pub static BOX_TEX_BYTES: &[u8] = include_bytes!("box.jpg");

/// DO NOT USE THE VERSION OF THIS FUNCTION THAT'S IN THE GFX-HAL CRATE.
///
/// It can trigger UB if you upcast from a low alignment to a higher alignment
/// type. You'll be sad.
pub fn cast_slice<T: Pod, U: Pod>(ts: &[T]) -> Option<&[U]> {
    use core::mem::{align_of, size_of};
    // Handle ZST (this all const folds)
    if size_of::<T>() == 0 || size_of::<U>() == 0 {
        if size_of::<T>() == size_of::<U>() {
            unsafe {
                return Some(core::slice::from_raw_parts(
                    ts.as_ptr() as *const U,
                    ts.len(),
                ));
            }
        } else {
            return None;
        }
    }
    // Handle alignments (this const folds)
    if align_of::<U>() > align_of::<T>() {
        // possible mis-alignment at the new type (this is a real runtime check)
        if (ts.as_ptr() as usize) % align_of::<U>() != 0 {
            return None;
        }
    }
    if size_of::<T>() == size_of::<U>() {
        // same size, so we direct cast, keeping the old length
        unsafe {
            Some(core::slice::from_raw_parts(
                ts.as_ptr() as *const U,
                ts.len(),
            ))
        }
    } else {
        // we might have slop, which would cause us to fail
        let byte_size = size_of::<T>() * ts.len();
        let (new_count, new_overflow) = (byte_size / size_of::<U>(), byte_size % size_of::<U>());
        if new_overflow > 0 {
            return None;
        } else {
            unsafe {
                Some(core::slice::from_raw_parts(
                    ts.as_ptr() as *const U,
                    new_count,
                ))
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct UniformBlock {
    projection: [[f32; 4]; 4],
}

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub position: glm::TVec3<f32>,
    pitch_deg: f32,
    yaw_deg: f32,
}

impl Camera {
    const UP: [f32; 3] = [0.0, 1.0, 0.0];

    fn make_front(&self) -> glm::TVec3<f32> {
        let pitch_rad = f32::to_radians(self.pitch_deg);
        let yaw_rad = f32::to_radians(self.yaw_deg);
        glm::make_vec3(&[
            yaw_rad.sin() * pitch_rad.cos(),
            pitch_rad.sin(),
            yaw_rad.cos() * pitch_rad.cos(),
        ])
    }

    pub fn update_orientation(&mut self, d_pitch_deg: f32, d_yaw_deg: f32) {
        self.pitch_deg = (self.pitch_deg + d_pitch_deg).max(-89.0).min(89.0);
        self.yaw_deg = (self.yaw_deg + d_yaw_deg) % 360.0;
    }

    pub fn update_position(&mut self, keys: &HashSet<VirtualKeyCode>, distance: f32) {
        let up = glm::make_vec3(&Self::UP);
        let forward = self.make_front();
        let cross_normalized = glm::cross::<f32, glm::U3>(&forward, &up).normalize();

        let mut move_vector = keys
            .iter()
            .fold(glm::make_vec3(&[0.0, 0.0, 0.0]), |vec, key| match *key {
                VirtualKeyCode::W => vec + forward,
                VirtualKeyCode::S => vec - forward,
                VirtualKeyCode::A => vec + cross_normalized,
                VirtualKeyCode::D => vec - cross_normalized,
                _ => vec,
            });

        if move_vector != glm::zero() {
            move_vector = move_vector.normalize();
            self.position += move_vector * distance;
        }
    }

    pub fn make_view_matrix(&self) -> glm::TMat4<f32> {
        glm::look_at_lh(
            &self.position,
            &(self.position + self.make_front()),
            &glm::make_vec3(&Self::UP),
        )
    }

    pub const fn at_position(position: glm::TVec3<f32>) -> Self {
        Self {
            position,
            pitch_deg: 0.0,
            yaw_deg: 0.0,
        }
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct Vertex {
    xyz: [f32; 3],
}
impl Vertex {
    pub fn attributes() -> Vec<AttributeDesc> {
        let position_attribute = AttributeDesc {
            location: 0,
            binding: 0,
            element: Element {
                format: Format::Rgb32Float,
                offset: 0,
            },
        };

        vec![position_attribute]
    }
}

const LINE_VERTICES: [Vertex; 4] = [
    Vertex {
        xyz: [-6.0, -4.0, 0.0],
    },
    Vertex {
        xyz: [-2.0, 4.0, 0.0],
    },
    Vertex {
        xyz: [3.0, -2.0, 0.0],
    },
    Vertex {
        xyz: [6.0, 3.0, 0.0],
    },
];

const LINE_INDICES: [u16; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

pub struct LoadedImage<B: Backend, D: Device<B>> {
    pub image: ManuallyDrop<B::Image>,
    pub requirements: Requirements,
    pub memory: ManuallyDrop<B::Memory>,
    pub image_view: ManuallyDrop<B::ImageView>,
    pub sampler: ManuallyDrop<B::Sampler>,
    pub phantom: PhantomData<D>,
}

impl<B: Backend, D: Device<B>> LoadedImage<B, D> {
    pub unsafe fn manually_drop(&self, device: &D) {
        use core::ptr::read;

        device.destroy_sampler(ManuallyDrop::into_inner(read(&self.sampler)));
        device.destroy_image_view(ManuallyDrop::into_inner(read(&self.image_view)));
        device.destroy_image(ManuallyDrop::into_inner(read(&self.image)));
        device.free_memory(ManuallyDrop::into_inner(read(&self.memory)));
    }

    pub fn new<C: Capability + Supports<Transfer>>(
        adapter: &Adapter<B>,
        device: &D,
        command_pool: &mut CommandPool<B, C>,
        command_queue: &mut CommandQueue<B, C>,
        img: image::RgbaImage,
    ) -> Result<Self, &'static str> {
        unsafe {
            // compute alignment sizes of image
            let pixel_size = size_of::<image::Rgba<u8>>();
            let row_size = pixel_size * (img.width() as usize);
            let limits = adapter.physical_device.limits();
            let row_alignment_mask = limits.min_buffer_copy_pitch_alignment as u32 - 1;
            let row_pitch = ((row_size as u32 + row_alignment_mask) & !row_alignment_mask) as usize;

            // Create a staging buffer to hold the memory for the image on the CPU VISIBLE memory
            let required_bytes = row_pitch * img.height() as usize;
            let staging_bundle =
                BufferBundle::new(&adapter, device, required_bytes, BufferUsage::TRANSFER_SRC)?;

            // Write image data to buffer
            let mut writer = device
                .acquire_mapping_writer(&staging_bundle.memory, 0..staging_bundle.requirements.size)
                .map_err(|_| "Could not acquire a mapping writer to the staging buffer")?;

            for y in 0..img.height() as usize {
                let row = &(*img)[y * row_size..(y + 1) * row_size];
                let dest_base = y * row_pitch;
                writer[dest_base..dest_base + row.len()].copy_from_slice(row);
            }

            device
                .release_mapping_writer(writer)
                .map_err(|_| "Could not release the mapping writer to the staging buffer")?;

            // Create description of image
            let mut image_description = device
                .create_image(
                    gfx_hal::image::Kind::D2(img.width(), img.height(), 1, 1),
                    1,
                    Format::Rgba8Srgb,
                    gfx_hal::image::Tiling::Optimal,
                    gfx_hal::image::Usage::TRANSFER_DST | gfx_hal::image::Usage::SAMPLED,
                    gfx_hal::image::ViewCapabilities::empty(),
                )
                .map_err(|_| "Could not create image description")?;

            // Allocate memory on GPU that is DEVICE_LOCAL
            let requirements = device.get_image_requirements(&image_description);
            let memory_type_id = adapter
                .physical_device
                .memory_properties()
                .memory_types
                .iter()
                .enumerate()
                .find(|&(id, memory_type)| {
                    requirements.type_mask & (1 << id) != 0
                        && memory_type.properties.contains(Properties::DEVICE_LOCAL)
                })
                .map(|(id, _)| MemoryTypeId(id))
                .ok_or("Could not find a memory type to support the image")?;

            let memory = device
                .allocate_memory(memory_type_id, requirements.size)
                .map_err(|_| "Could not allocate memory")?;

            device
                .bind_image_memory(&memory, 0, &mut image_description)
                .map_err(|_| "Could not bind the image memory")?;

            // Create image view and sampler
            let image_view = device
                .create_image_view(
                    &image_description,
                    gfx_hal::image::ViewKind::D2,
                    Format::Rgba8Srgb,
                    gfx_hal::format::Swizzle::NO,
                    SubresourceRange {
                        aspects: Aspects::COLOR,
                        levels: 0..1,
                        layers: 0..1,
                    },
                )
                .map_err(|_| "Could not create image view")?;

            let sampler = device
                .create_sampler(gfx_hal::image::SamplerInfo::new(
                    gfx_hal::image::Filter::Nearest,
                    gfx_hal::image::WrapMode::Tile,
                ))
                .map_err(|_| "Could not create sampler")?;

            // Create command buffer
            let mut cmd_buffer = command_pool.acquire_command_buffer::<gfx_hal::command::OneShot>();
            cmd_buffer.begin();

            // Tell GPU to set memory/image to TRANSER_WRITE state
            let image_barrier = gfx_hal::memory::Barrier::Image {
                states: (gfx_hal::image::Access::empty(), Layout::Undefined)
                    ..(
                        gfx_hal::image::Access::TRANSFER_WRITE,
                        Layout::TransferDstOptimal,
                    ),
                target: &image_description,
                families: None,
                range: SubresourceRange {
                    aspects: Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };
            cmd_buffer.pipeline_barrier(
                PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER,
                gfx_hal::memory::Dependencies::empty(),
                &[image_barrier],
            );

            // Copy data from staging buffer (CPU_VISIBLE) to the GPU (DEVICE_LOCAL)
            cmd_buffer.copy_buffer_to_image(
                &staging_bundle.buffer,
                &image_description,
                Layout::TransferDstOptimal,
                &[gfx_hal::command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: (row_pitch / pixel_size) as u32,
                    buffer_height: img.height(),
                    image_layers: SubresourceLayers {
                        aspects: Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: gfx_hal::image::Offset { x: 0, y: 0, z: 0 },
                    image_extent: Extent {
                        width: img.width(),
                        height: img.height(),
                        depth: 1,
                    },
                }],
            );

            // Tell GPU to set memory/image to SHADER_READ state
            let image_barrier = gfx_hal::memory::Barrier::Image {
                states: (
                    gfx_hal::image::Access::TRANSFER_WRITE,
                    Layout::TransferDstOptimal,
                )
                    ..(
                        gfx_hal::image::Access::SHADER_READ,
                        Layout::ShaderReadOnlyOptimal,
                    ),
                target: &image_description,
                families: None,
                range: SubresourceRange {
                    aspects: Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };

            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
                gfx_hal::memory::Dependencies::empty(),
                &[image_barrier],
            );

            cmd_buffer.finish();
            let upload_fence = device
                .create_fence(false)
                .map_err(|_| "Could not create upload fence")?;
            command_queue.submit_nosemaphores(Some(&cmd_buffer), Some(&upload_fence));
            device
                .wait_for_fence(&upload_fence, core::u64::MAX)
                .map_err(|_| "Could not wait for the upload fence")?;
            device.destroy_fence(upload_fence);

            // Destroy staging bundle and command buffer
            staging_bundle.manually_drop(device);
            command_pool.free(Some(cmd_buffer));

            Ok(Self {
                image: ManuallyDrop::new(image_description),
                requirements,
                memory: ManuallyDrop::new(memory),
                image_view: ManuallyDrop::new(image_view),
                sampler: ManuallyDrop::new(sampler),
                phantom: PhantomData,
            })
        }
    }
}

pub struct BufferBundle<B: Backend, D: Device<B>> {
    pub buffer: ManuallyDrop<B::Buffer>,
    pub requirements: Requirements,
    pub memory: ManuallyDrop<B::Memory>,
    pub phantom: PhantomData<D>,
}

impl<B: Backend, D: Device<B>> BufferBundle<B, D> {
    pub fn new(
        adapter: &Adapter<B>,
        device: &D,
        size: usize,
        usage: BufferUsage,
    ) -> Result<Self, &'static str> {
        unsafe {
            let (buffer, memory, requirements) = unsafe {
                let mut buffer = device
                    .create_buffer(size as u64, usage)
                    .map_err(|_| "Could not create a buffer for the vertices")?;

                let requirements = device.get_buffer_requirements(&buffer);
                let memory_type_id = adapter
                    .physical_device
                    .memory_properties()
                    .memory_types
                    .iter()
                    .enumerate()
                    .find(|&(id, memory_type)| {
                        requirements.type_mask & (1 << id) != 0
                            && memory_type.properties.contains(Properties::CPU_VISIBLE)
                    })
                    .map(|(id, _)| MemoryTypeId(id))
                    .ok_or("Could not find a memory type to support the vertex buffer")?;

                let memory = device
                    .allocate_memory(memory_type_id, requirements.size)
                    .map_err(|_| "Could not allocate memory")?;

                device
                    .bind_buffer_memory(&memory, 0, &mut buffer)
                    .map_err(|_| "Could not bind the buffer memory")?;

                (buffer, memory, requirements)
            };

            Ok(Self {
                buffer: ManuallyDrop::new(buffer),
                requirements,
                memory: ManuallyDrop::new(memory),
                phantom: PhantomData,
            })
        }
    }

    pub unsafe fn manually_drop(&self, device: &D) {
        use core::ptr::read;
        device.destroy_buffer(ManuallyDrop::into_inner(read(&self.buffer)));
        device.free_memory(ManuallyDrop::into_inner(read(&self.memory)));
    }
}

#[derive(Debug)]
pub struct WinitState {
    pub events_loop: EventsLoop,
    pub window: Window,
    pub keys_held: HashSet<VirtualKeyCode>,
}

impl WinitState {
    pub fn new<T: Into<String>>(title: T, size: LogicalSize) -> Result<Self, CreationError> {
        let events_loop = EventsLoop::new();
        let output = WindowBuilder::new()
            .with_title(title)
            .with_dimensions(size)
            .build(&events_loop);

        output.map(|window| Self {
            events_loop,
            window,
            keys_held: HashSet::new(),
        })
    }
}

const WINDOW_NAME: &str = "Hello World";

impl Default for WinitState {
    fn default() -> Self {
        Self::new(
            WINDOW_NAME,
            LogicalSize {
                width: 800.0,
                height: 600.0,
            },
        )
        .expect("Could not create window")
    }
}

pub struct HalState {
    creation_instant: Instant,
    cube_vertices: BufferBundle<back::Backend, back::Device>,
    cube_indexes: BufferBundle<back::Backend, back::Device>,
    uniform: BufferBundle<back::Backend, back::Device>,
    descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    descriptor_pool: ManuallyDrop<<back::Backend as Backend>::DescriptorPool>,
    descriptor_set: ManuallyDrop<<back::Backend as Backend>::DescriptorSet>,
    pipeline_layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
    graphics_pipeline: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,
    current_frame: usize,
    frames_in_flight: usize,
    in_flight_fences: Vec<<back::Backend as Backend>::Fence>,
    render_finished_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    image_available_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    command_buffers: Vec<CommandBuffer<back::Backend, Graphics, MultiShot, Primary>>,
    command_pool: ManuallyDrop<CommandPool<back::Backend, Graphics>>,
    framebuffers: Vec<<back::Backend as Backend>::Framebuffer>,
    image_views: Vec<(<back::Backend as Backend>::ImageView)>,
    render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
    render_area: Rect,
    queue_group: QueueGroup<back::Backend, Graphics>,
    swapchain: ManuallyDrop<<back::Backend as Backend>::Swapchain>,
    device: ManuallyDrop<back::Device>,
    _adapter: Adapter<back::Backend>,
    _surface: <back::Backend as Backend>::Surface,
    _instance: ManuallyDrop<back::Instance>,
}

impl HalState {
    pub fn new(window: &Window) -> Result<Self, &'static str> {
        let instance = back::Instance::create(WINDOW_NAME, 1);
        let mut surface = instance.create_surface(window);

        let adapter = instance
            .enumerate_adapters()
            .into_iter()
            .find(|a| {
                a.queue_families
                    .iter()
                    .any(|qf| qf.supports_graphics() && surface.supports_queue_family(qf))
            })
            .ok_or("Could not find a graphical adapter")?;

        let (mut device, mut queue_group) = {
            let queu_family = adapter
                .queue_families
                .iter()
                .find(|qf| qf.supports_graphics() && surface.supports_queue_family(qf))
                .ok_or("Could not find a QueueFamily with graphics")?;

            let Gpu { device, mut queues } = unsafe {
                adapter
                    .physical_device
                    .open(&[(&queu_family, &[1.0; 1])])
                    .map_err(|_| "Could not open the PhysicalDevice")?
            };

            let queue_group = queues
                .take::<Graphics>(queu_family.id())
                .ok_or("Couldn't take ownership of the QueueGroup!")?;

            let _ = if queue_group.queues.len() > 0 {
                Ok(())
            } else {
                Err("The QueueGroup did not have any CommandQueues available")
            }?;

            (device, queue_group)
        };

        let (swapchain, extent, backbuffer, format, frames_in_flight) = {
            let (caps, preferred_formats, present_modes, composite_alphas) =
                surface.compatibility(&adapter.physical_device);

            info!("{:?}", caps);
            info!("Preferred Formats: {:?}", preferred_formats);
            info!("Present Modes: {:?}", present_modes);
            info!("Composite Alphas: {:?}", composite_alphas);

            let present_mode = {
                use gfx_hal::window::PresentMode::*;

                [Mailbox, Fifo, Relaxed, Immediate]
                    .iter()
                    .cloned()
                    .find(|pm| present_modes.contains(pm))
                    .ok_or("No present mode found")?
            };

            let composite_alpha = {
                use gfx_hal::window::CompositeAlpha::*;
                [Opaque, Inherit, PreMultiplied, PostMultiplied]
                    .iter()
                    .cloned()
                    .find(|ca| composite_alphas.contains(ca))
                    .ok_or("No composite alpha found")?
            };

            let format = match preferred_formats {
                None => Format::Rgba8Srgb,
                Some(formats) => match formats
                    .iter()
                    .find(|format| format.base_format().1 == ChannelType::Srgb)
                    .cloned()
                {
                    Some(srgb_format) => srgb_format,
                    None => formats
                        .get(0)
                        .cloned()
                        .ok_or("Preferred format list was empty")?,
                },
            };

            let extent = {
                let window_client_area = window
                    .get_inner_size()
                    .ok_or("Window does not exist")?
                    .to_physical(window.get_hidpi_factor());
                Extent2D {
                    width: caps.extents.end.width.min(window_client_area.width as u32),
                    height: caps
                        .extents
                        .end
                        .height
                        .min(window_client_area.height as u32),
                }
            };

            let image_count = if present_mode == PresentMode::Mailbox {
                (caps.image_count.end - 1).min(3)
            } else {
                (caps.image_count.end - 1).min(2)
            };
            let image_layers = 1;
            let image_usage = if caps.usage.contains(Usage::COLOR_ATTACHMENT) {
                Usage::COLOR_ATTACHMENT
            } else {
                Err("The surface isn't capable of support color")?
            };

            let swapchain_config = SwapchainConfig {
                present_mode,
                composite_alpha,
                format,
                extent,
                image_count,
                image_layers,
                image_usage,
            };

            info!("Swapchain Config {:?}", swapchain_config);

            let (swapchain, backbuffer) = unsafe {
                device
                    .create_swapchain(&mut surface, swapchain_config, None)
                    .map_err(|_| "Failed to create swapchain")?
            };

            (swapchain, extent, backbuffer, format, image_count as usize)
        };

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) = {
            let mut image_available_semaphores: Vec<<back::Backend as Backend>::Semaphore> = vec![];
            let mut render_finished_semaphores: Vec<<back::Backend as Backend>::Semaphore> = vec![];
            let mut in_flight_fences: Vec<<back::Backend as Backend>::Fence> = vec![];

            for _ in 0..frames_in_flight {
                image_available_semaphores.push(
                    device
                        .create_semaphore()
                        .map_err(|_| "Could not create semaphore")?,
                );
                render_finished_semaphores.push(
                    device
                        .create_semaphore()
                        .map_err(|_| "Could not create semaphore")?,
                );
                in_flight_fences.push(
                    device
                        .create_fence(true)
                        .map_err(|_| "Could not create fence")?,
                );
            }
            (
                image_available_semaphores,
                render_finished_semaphores,
                in_flight_fences,
            )
        };

        let render_pass = {
            let color_attachment = Attachment {
                format: Some(format),
                samples: 1,
                ops: AttachmentOps {
                    load: AttachmentLoadOp::Clear,
                    store: AttachmentStoreOp::Store,
                },
                stencil_ops: AttachmentOps::DONT_CARE,
                layouts: Layout::Undefined..Layout::Present,
            };
            let subpass = SubpassDesc {
                colors: &[(0, Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            unsafe {
                device
                    .create_render_pass(&[color_attachment], &[subpass], &[])
                    .map_err(|_| "Could not create renderpass")?
            }
        };

        let image_views: Vec<_> = match backbuffer {
            Backbuffer::Images(images) => images
                .into_iter()
                .map(|image| unsafe {
                    device
                        .create_image_view(
                            &image,
                            ViewKind::D2,
                            format,
                            Swizzle::NO,
                            SubresourceRange {
                                aspects: Aspects::COLOR,
                                levels: 0..1,
                                layers: 0..1,
                            },
                        )
                        .map_err(|_| "Could not create image_view")
                })
                .collect::<Result<Vec<_>, &str>>()?,
            Backbuffer::Framebuffer(_) => unimplemented!("Can not handle framebuffer backbuffer"),
        };

        let framebuffers: Vec<<back::Backend as Backend>::Framebuffer> = {
            image_views
                .iter()
                .map(|image_view| unsafe {
                    device
                        .create_framebuffer(
                            &render_pass,
                            vec![image_view],
                            Extent {
                                width: extent.width as u32,
                                height: extent.height as u32,
                                depth: 1,
                            },
                        )
                        .map_err(|_| "Failed to create a framebuffer")
                })
                .collect::<Result<Vec<_>, &str>>()?
        };

        let mut command_pool = unsafe {
            device
                .create_command_pool_typed(&queue_group, CommandPoolCreateFlags::RESET_INDIVIDUAL)
                .map_err(|_| "Could not create the raw command pool")?
        };

        let command_buffers: Vec<_> = framebuffers
            .iter()
            .map(|_| command_pool.acquire_command_buffer())
            .collect();

        let (
            descriptor_set_layouts,
            descriptor_set,
            descriptor_pool,
            pipeline_layout,
            graphics_pipeline,
        ) = Self::create_pipeline(&mut device, extent, &render_pass)?;

        let cubic = lyon::geom::CubicBezierSegment {
            from: lyon::math::point(LINE_VERTICES[0].xyz[0], LINE_VERTICES[0].xyz[1]),
            ctrl1: lyon::math::point(LINE_VERTICES[1].xyz[0], LINE_VERTICES[1].xyz[1]),
            ctrl2: lyon::math::point(LINE_VERTICES[2].xyz[0], LINE_VERTICES[2].xyz[1]),
            to: lyon::math::point(LINE_VERTICES[3].xyz[0], LINE_VERTICES[3].xyz[1]),
        };

        let mut line_triangle: Vec<_> = cubic.flattened(0.05).collect();

        // Add first point
        line_triangle.insert(
            0,
            cubic.from
        );
        line_triangle.push(cubic.to);

        // Create small triangle on every LINE_VERTEX
        let mut line_triangle: Vec<Vertex> = line_triangle
            .into_iter()
            .flat_map(|a| {
                let lyon::math::Point { x, y, .. } = a;

                vec![
                    Vertex {
                        xyz: [x, y + 0.1, 0.0],
                    },
                    Vertex {
                        xyz: [x + 0.1, y - 0.1, 0.0],
                    },
                    Vertex {
                        xyz: [x - 0.1, y - 0.1, 0.0],
                    },
                ]
            })
            .collect();

        let vertices = BufferBundle::new(
            &adapter,
            &device,
            size_of::<Vertex>() * &line_triangle.len(),
            BufferUsage::VERTEX,
        )?;

        // Write the vertex data just once
        unsafe {
            let mut data_target = device
                .acquire_mapping_writer(&vertices.memory, 0..vertices.requirements.size)
                .map_err(|_| "Failed to acquire an vertex buffer mapping writer")?;
            data_target[..line_triangle.len()].copy_from_slice(&line_triangle);
            device
                .release_mapping_writer(data_target)
                .map_err(|_| "Could not release the vertex buffer mapping writer")?;
        }
        let triangle_indices = &line_triangle.len() * 3;

        let indexes = BufferBundle::new(
            &adapter,
            &device,
            size_of::<u16>() * triangle_indices,
            BufferUsage::INDEX,
        )?;

        let mut triangle_slice: Vec<u16> = vec![];
        for i in 0..triangle_indices {
            triangle_slice.push(i as u16);
        }

        // Write the indices of the quad only once
        unsafe {
            let mut data_target = device
                .acquire_mapping_writer(&indexes.memory, 0..indexes.requirements.size)
                .map_err(|_| "Failed to acquire an index buffer mapping writer")?;

            data_target[..triangle_indices].copy_from_slice(&triangle_slice);
            device
                .release_mapping_writer(data_target)
                .map_err(|_| "Could not release the index buffer mapping writer")?;
        }

        // Create uniform buffer
        let uniform = BufferBundle::new(
            &adapter,
            &device,
            size_of::<UniformBlock>(),
            BufferUsage::UNIFORM,
        )?;

        const UNIFORM_DATA: [UniformBlock; 1] = [UniformBlock {
            projection: [
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }];

        // Write default data to uniform buffer
        unsafe {
            let mut data_target = device
                .acquire_mapping_writer(&uniform.memory, 0..uniform.requirements.size)
                .map_err(|_| "Could not acquire a uniform buffer mapping writer")?;

            data_target[..UNIFORM_DATA.len()].copy_from_slice(&UNIFORM_DATA);

            device
                .release_mapping_writer(data_target)
                .map_err(|_| "Could not release the uniform buffer mapping writer")?;
        }

        // Create descriptors
        let texture = LoadedImage::new(
            &adapter,
            &device,
            &mut command_pool,
            &mut queue_group.queues[0],
            image::load_from_memory(BOX_TEX_BYTES)
                .expect("Binary corrupted")
                .to_rgba(),
        )?;

        unsafe {
            device.write_descriptor_sets(vec![
                gfx_hal::pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(gfx_hal::pso::Descriptor::Image(
                        texture.image_view.deref(),
                        Layout::Undefined,
                    )),
                },
                gfx_hal::pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 1,
                    array_offset: 0,
                    descriptors: Some(gfx_hal::pso::Descriptor::Sampler(texture.sampler.deref())),
                },
                gfx_hal::pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 2,
                    array_offset: 0,
                    descriptors: Some(gfx_hal::pso::Descriptor::Buffer(
                        uniform.buffer.deref(),
                        None..None,
                    )),
                },
            ])
        }

        let creation_instant = Instant::now();

        Ok(Self {
            creation_instant,
            cube_indexes: indexes,
            cube_vertices: vertices,
            uniform,
            descriptor_set_layouts,
            descriptor_set: ManuallyDrop::new(descriptor_set),
            descriptor_pool: ManuallyDrop::new(descriptor_pool),
            pipeline_layout: ManuallyDrop::new(pipeline_layout),
            graphics_pipeline: ManuallyDrop::new(graphics_pipeline),
            _instance: ManuallyDrop::new(instance),
            _surface: surface,
            _adapter: adapter,
            device: ManuallyDrop::new(device),
            queue_group,
            swapchain: ManuallyDrop::new(swapchain),
            render_area: extent.to_extent().rect(),
            render_pass: ManuallyDrop::new(render_pass),
            image_views,
            framebuffers,
            command_pool: ManuallyDrop::new(command_pool),
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            frames_in_flight,
            current_frame: 0,
        })
    }
    fn create_pipeline(
        device: &mut back::Device,
        extent: Extent2D,
        render_pass: &<back::Backend as Backend>::RenderPass,
    ) -> Result<
        (
            Vec<<back::Backend as Backend>::DescriptorSetLayout>,
            <back::Backend as Backend>::DescriptorSet,
            <back::Backend as Backend>::DescriptorPool,
            <back::Backend as Backend>::PipelineLayout,
            <back::Backend as Backend>::GraphicsPipeline,
        ),
        &'static str,
    > {
        let mut compiler = shaderc::Compiler::new().ok_or("shaderc not found")?;

        let vertex_compile_artifact = compiler
            .compile_into_spirv(
                VERTEX_SOURCE,
                shaderc::ShaderKind::Vertex,
                "vertex.vert",
                "main",
                None,
            )
            .map_err(|_| "Could not compile vertex shader")?;
        let fragment_compile_artifact = compiler
            .compile_into_spirv(
                FRAGMENT_SOURCE,
                shaderc::ShaderKind::Fragment,
                "fragment.frag",
                "main",
                None,
            )
            .map_err(|e| {
                debug!("{:?}", e);
                return "Could not compile fragment shader";
            })?;
        let vertex_shader_module = unsafe {
            device
                .create_shader_module(vertex_compile_artifact.as_binary_u8())
                .map_err(|_| "Could not make the vertex module")?
        };
        let fragment_shader_module = unsafe {
            device
                .create_shader_module(fragment_compile_artifact.as_binary_u8())
                .map_err(|_| "Could not make the fragment module")?
        };
        let (descriptor_set_layouts, descriptor_pool, descriptor_set, layout, gfx_pipeline) = {
            let (vs_entry, fs_entry) = (
                EntryPoint {
                    entry: "main",
                    module: &vertex_shader_module,
                    specialization: Specialization {
                        constants: &[],
                        data: &[],
                    },
                },
                EntryPoint {
                    entry: "main",
                    module: &fragment_shader_module,
                    specialization: Specialization {
                        constants: &[],
                        data: &[],
                    },
                },
            );

            let shader_set = GraphicsShaderSet {
                vertex: vs_entry,
                hull: None,
                domain: None,
                geometry: None,
                fragment: Some(fs_entry),
            };

            let vertex_buffers: Vec<VertexBufferDesc> = vec![VertexBufferDesc {
                binding: 0,
                stride: (size_of::<Vertex>()) as ElemStride,
                rate: 0,
            }];

            let attributes: Vec<AttributeDesc> = Vertex::attributes();

            let rasterizer = Rasterizer {
                polygon_mode: PolygonMode::Line(1.0),
                cull_face: Face::BACK,
                front_face: FrontFace::Clockwise,
                depth_clamping: false,
                depth_bias: None,
                conservative: false,
            };
            let depth_stencil = DepthStencilDesc {
                depth: DepthTest::Off,
                depth_bounds: false,
                stencil: StencilTest::Off,
            };

            let blender = {
                let blend_state = BlendState::On {
                    color: BlendOp::Add {
                        src: Factor::One,
                        dst: Factor::Zero,
                    },
                    alpha: BlendOp::Add {
                        src: Factor::One,
                        dst: Factor::Zero,
                    },
                };
                BlendDesc {
                    logic_op: Some(LogicOp::Copy),
                    targets: vec![ColorBlendDesc(ColorMask::ALL, blend_state)],
                }
            };

            let baked_states = BakedStates {
                viewport: Some(Viewport {
                    rect: extent.to_extent().rect(),
                    depth: (0.0..1.0),
                }),
                scissor: Some(extent.to_extent().rect()),
                blend_color: None,
                depth_bounds: None,
            };

            let bindings = &[
                DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: gfx_hal::pso::DescriptorType::SampledImage,
                    count: 1,
                    stage_flags: ShaderStageFlags::FRAGMENT,
                    immutable_samplers: false,
                },
                DescriptorSetLayoutBinding {
                    binding: 1,
                    ty: gfx_hal::pso::DescriptorType::Sampler,
                    count: 1,
                    stage_flags: ShaderStageFlags::FRAGMENT,
                    immutable_samplers: false,
                },
                DescriptorSetLayoutBinding {
                    binding: 2,
                    ty: gfx_hal::pso::DescriptorType::UniformBuffer,
                    count: 1,
                    stage_flags: ShaderStageFlags::VERTEX,
                    immutable_samplers: false,
                },
            ];
            let immutable_samplers = Vec::<<back::Backend as Backend>::Sampler>::new();
            let descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout> =
                vec![unsafe {
                    device
                        .create_descriptor_set_layout(bindings, immutable_samplers)
                        .map_err(|_| "Could not make a descriptor set layout")?
                }];
            let push_constants = vec![(ShaderStageFlags::VERTEX, 0..16)];

            // Create a descriptor pool
            let mut descriptor_pool = unsafe {
                device
                    .create_descriptor_pool(
                        1,
                        &[
                            gfx_hal::pso::DescriptorRangeDesc {
                                ty: gfx_hal::pso::DescriptorType::SampledImage,
                                count: 1,
                            },
                            gfx_hal::pso::DescriptorRangeDesc {
                                ty: gfx_hal::pso::DescriptorType::Sampler,
                                count: 1,
                            },
                            gfx_hal::pso::DescriptorRangeDesc {
                                ty: gfx_hal::pso::DescriptorType::UniformBuffer,
                                count: 1,
                            },
                        ],
                    )
                    .map_err(|_| "Could not create a descriptor pool")?
            };

            let descriptor_set = unsafe {
                descriptor_pool
                    .allocate_set(&descriptor_set_layouts[0])
                    .map_err(|_| "Could not create descriptor set")?
            };

            let layout = unsafe {
                device
                    .create_pipeline_layout(&descriptor_set_layouts, push_constants)
                    .map_err(|_| "Could not create a pipeline layout")?
            };
            let gfx_pipeline = {
                let desc = {
                    GraphicsPipelineDesc {
                        shaders: shader_set,
                        rasterizer,
                        vertex_buffers,
                        attributes,
                        input_assembler: InputAssemblerDesc::new(Primitive::TriangleList),
                        blender,
                        depth_stencil,
                        multisampling: None,
                        baked_states,
                        layout: &layout,
                        subpass: Subpass {
                            index: 0,
                            main_pass: render_pass,
                        },
                        flags: PipelineCreationFlags::empty(),
                        parent: BasePipeline::None,
                    }
                };

                unsafe {
                    device
                        .create_graphics_pipeline(&desc, None)
                        .map_err(|_| "Could not create a graphics pipeline")?
                }
            };

            (
                descriptor_set_layouts,
                descriptor_pool,
                descriptor_set,
                layout,
                gfx_pipeline,
            )
        };
        unsafe {
            device.destroy_shader_module(vertex_shader_module);
            device.destroy_shader_module(fragment_shader_module);
        }

        Ok((
            descriptor_set_layouts,
            descriptor_set,
            descriptor_pool,
            layout,
            gfx_pipeline,
        ))
    }

    pub fn draw_cubes_frame(
        &mut self,
        view_projection: &glm::TMat4<f32>,
        models: &[glm::TMat4<f32>],
    ) -> Result<(), &'static str> {
        // Setup for this frame
        let image_available = &self.image_available_semaphores[self.current_frame];
        let render_finished = &self.render_finished_semaphores[self.current_frame];

        // Advance the frame before we extract optionals
        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;

        let (i_u32, i_usize) = unsafe {
            let image_index = self
                .swapchain
                .acquire_image(core::u64::MAX, FrameSync::Semaphore(image_available))
                .expect("Couldn't acquire an image from the swapchain!");

            (image_index, image_index as usize)
        };

        let flight_fence = &self.in_flight_fences[i_usize];
        unsafe {
            self.device
                .wait_for_fence(flight_fence, core::u64::MAX)
                .map_err(|_| "Failed to wait on flight_fence")?;

            self.device
                .reset_fence(flight_fence)
                .map_err(|_| "Couln't reset flight_fence")?;
        }

        // Time data
        let duration = Instant::now().duration_since(self.creation_instant);
        let time_f32 = duration.as_secs() as f32 + duration.subsec_nanos() as f32 * 1e-9;
        // Record Commands
        unsafe {
            let buffer = &mut self.command_buffers[i_usize];
            const TRIANGLE_CLEAR: [ClearValue; 1] =
                [ClearValue::Color(ClearColor::Float([0.1, 0.2, 0.3, 1.0]))];
            buffer.begin(false);
            {
                let mut encoder = buffer.begin_render_pass_inline(
                    &self.render_pass,
                    &self.framebuffers[i_usize],
                    self.render_area,
                    TRIANGLE_CLEAR.iter(),
                );

                encoder.bind_graphics_pipeline(&self.graphics_pipeline);

                let vertex_buffers: ArrayVec<[_; 1]> =
                    [(self.cube_vertices.buffer.deref(), 0)].into();
                encoder.bind_vertex_buffers(0, vertex_buffers);
                encoder.bind_index_buffer(IndexBufferView {
                    buffer: &self.cube_indexes.buffer,
                    offset: 0,
                    index_type: IndexType::U16,
                });

                encoder.bind_graphics_descriptor_sets(
                    &self.pipeline_layout,
                    0,
                    Some(self.descriptor_set.deref()),
                    &[],
                );

                for model in models.iter() {
                    let mvp = view_projection * model;
                    encoder.push_graphics_constants(
                        &self.pipeline_layout,
                        ShaderStageFlags::VERTEX,
                        0,
                        cast_slice::<f32, u32>(&mvp.data).expect("This never fails"),
                    );

                    encoder.draw_indexed(0..36, 0, 0..1);
                }
            }
            buffer.finish();
        }

        // Submission and Present
        let command_buffers = &self.command_buffers[i_usize..=i_usize];
        let wait_semaphores: ArrayVec<[_; 1]> =
            [(image_available, PipelineStage::COLOR_ATTACHMENT_OUTPUT)].into();
        let signal_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let present_wait_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let submission = Submission {
            command_buffers,
            wait_semaphores,
            signal_semaphores,
        };
        let the_command_queue = &mut self.queue_group.queues[0];
        unsafe {
            the_command_queue.submit(submission, Some(flight_fence));
            self.swapchain
                .present(the_command_queue, i_u32, present_wait_semaphores)
                .map_err(|_| "Failed to present into the swapchain")
        }
    }

    pub fn draw_clear_frame(&mut self, color: [f32; 4]) -> Result<(), &'static str> {
        // Setup for this frame
        let image_available = &self.image_available_semaphores[self.current_frame];
        let render_finished = &self.render_finished_semaphores[self.current_frame];

        // Advance the frame before we extract optionals
        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;

        let (i_u32, i_usize) = unsafe {
            let image_index = self
                .swapchain
                .acquire_image(core::u64::MAX, FrameSync::Semaphore(image_available))
                .expect("Couldn't acquire an image from the swapchain!");

            (image_index, image_index as usize)
        };

        let flight_fence = &self.in_flight_fences[i_usize];
        unsafe {
            self.device
                .wait_for_fence(flight_fence, core::u64::MAX)
                .map_err(|_| "Failed to wait on flight_fence")?;

            self.device
                .reset_fence(flight_fence)
                .map_err(|_| "Couln't reset flight_fence")?;
        }

        // Record Commands
        unsafe {
            let buffer = &mut self.command_buffers[i_usize];
            let clear_values = [ClearValue::Color(ClearColor::Float(color))];

            buffer.begin(false);
            buffer.begin_render_pass_inline(
                &self.render_pass,
                &self.framebuffers[i_usize],
                self.render_area,
                clear_values.iter(),
            );

            buffer.finish();
        }

        // Submission and Present
        let command_buffers = &self.command_buffers[i_usize..=i_usize];
        let wait_semaphores: ArrayVec<[_; 1]> =
            [(image_available, PipelineStage::COLOR_ATTACHMENT_OUTPUT)].into();
        let signal_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let present_wait_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let submission = Submission {
            command_buffers,
            wait_semaphores,
            signal_semaphores,
        };
        let the_command_queue = &mut self.queue_group.queues[0];
        unsafe {
            the_command_queue.submit(submission, Some(flight_fence));
            self.swapchain
                .present(the_command_queue, i_u32, present_wait_semaphores)
                .map_err(|_| "Failed to present into the swapchain")
        }
    }
}

impl core::ops::Drop for HalState {
    fn drop(&mut self) {
        let _ = self.device.wait_idle();

        unsafe {
            for fence in self.in_flight_fences.drain(..) {
                self.device.destroy_fence(fence)
            }
            for semaphore in self.render_finished_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore)
            }
            for semaphore in self.image_available_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore)
            }
            for framebuffer in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(framebuffer);
            }
            for image_view in self.image_views.drain(..) {
                self.device.destroy_image_view(image_view);
            }
            // LAST RESORT STYLE CODE, NOT TO BE IMITATED LIGHTLY
            use core::ptr::read;
            self.device.destroy_command_pool(
                ManuallyDrop::into_inner(read(&self.command_pool)).into_raw(),
            );
            self.device
                .destroy_render_pass(ManuallyDrop::into_inner(read(&self.render_pass)));
            self.device
                .destroy_swapchain(ManuallyDrop::into_inner(read(&self.swapchain)));
            ManuallyDrop::drop(&mut self.device);
            ManuallyDrop::drop(&mut self._instance);
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct UserInput {
    pub end_requested: bool,
    pub new_frame_size: Option<(f64, f64)>,
    pub new_mouse_position: Option<(f64, f64)>,
    pub seconds: f32,
    pub keys_held: HashSet<VirtualKeyCode>,
}

impl UserInput {
    pub fn poll_events_loop(winit_state: &mut WinitState, last_timestamp: &mut Instant) -> Self {
        let mut output = UserInput::default();
        let events_loop = &mut winit_state.events_loop;
        let keys_held = &mut winit_state.keys_held;

        events_loop.poll_events(|event| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => output.end_requested = true,
            Event::DeviceEvent {
                event:
                    DeviceEvent::Key(KeyboardInput {
                        virtual_keycode: Some(code),
                        state,
                        ..
                    }),
                ..
            } => drop(match state {
                ElementState::Pressed => keys_held.insert(code),
                ElementState::Released => keys_held.remove(&code),
            }),
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state,
                                virtual_keycode: Some(code),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                #[cfg(feature = "metal")]
                {
                    match state {
                        ElementState::Pressed => keys_held.insert(code),
                        ElementState::Released => keys_held.remove(&code),
                    }
                };
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(logical),
                ..
            } => {
                output.new_frame_size = Some((logical.width, logical.height));
            }
            _ => (),
        });

        output.seconds = {
            let now = Instant::now();
            let duration = now.duration_since(*last_timestamp);
            *last_timestamp = now;
            duration.as_secs() as f32 + duration.subsec_nanos() as f32 * 1e-9
        };
        output.keys_held = keys_held.clone();

        output
    }
}

#[derive(Debug, Clone)]
pub struct LocalState {
    pub frame_width: f64,
    pub frame_height: f64,
    pub mouse_x: f64,
    pub mouse_y: f64,
    pub camera: Camera,
    pub projection: glm::TMat4<f32>,
    pub cubes: Vec<glm::TMat4<f32>>,
    pub spare_time: f32,
}

impl LocalState {
    pub fn update_from_input(&mut self, input: UserInput) {
        if let Some(frame_size) = input.new_frame_size {
            self.frame_width = frame_size.0;
            self.frame_height = frame_size.1;
        }
        if let Some(position) = input.new_mouse_position {
            self.mouse_x = position.0;
            self.mouse_y = position.1;
        }
        assert!(self.frame_width != 0.0 && self.frame_height != 0.0);

        self.spare_time += input.seconds;

        self.camera.update_position(&input.keys_held, 0.1);
    }
}

fn do_the_render(hal_state: &mut HalState, local_state: &LocalState) -> Result<(), &'static str> {
    let vp = local_state.projection * local_state.camera.make_view_matrix();

    // Update uniform buffer

    let duration = Instant::now().duration_since(hal_state.creation_instant);
    let time_f32 = duration.as_secs() as f32 + duration.subsec_nanos() as f32 * 1e-9;

    hal_state.draw_cubes_frame(&vp, &local_state.cubes)
}

fn main() {
    simple_logger::init().unwrap();

    let mut winit_state = WinitState::default();

    let mut hal_state = match HalState::new(&winit_state.window) {
        Ok(state) => state,
        Err(e) => panic!(e),
    };
    let (frame_width, frame_height) = winit_state
        .window
        .get_inner_size()
        .map(|logical| logical.into())
        .unwrap_or((0.0, 0.0));

    let mut local_state = LocalState {
        frame_width,
        frame_height,
        mouse_x: 0.0,
        mouse_y: 0.0,
        camera: Camera::at_position(glm::make_vec3(&[0.0, 0.0, -5.0])),
        projection: {
            let mut temp = glm::perspective_lh_zo(800.0 / 600.0, f32::to_radians(50.0), 0.1, 100.0);
            temp[(1, 1)] *= -1.0;
            temp
        },
        cubes: vec![glm::identity()],
        spare_time: 0.0,
    };
    let mut last_timestamp = Instant::now();
    loop {
        let inputs = UserInput::poll_events_loop(&mut winit_state, &mut last_timestamp);
        if inputs.end_requested {
            break;
        }

        if inputs.new_frame_size.is_some() {
            debug!("Resized window, recreating HalState");
            drop(hal_state);
            hal_state = match HalState::new(&winit_state.window) {
                Ok(state) => state,
                Err(e) => panic!(e),
            };
        }

        local_state.update_from_input(inputs);
        if let Err(e) = do_the_render(&mut hal_state, &local_state) {
            error!("Rendering Error: {:?}", e);
            debug!("Resized window, recreating HalState");
            drop(hal_state);
            hal_state = match HalState::new(&winit_state.window) {
                Ok(state) => state,
                Err(e) => panic!(e),
            };
        }
    }
}
