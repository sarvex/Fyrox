use crate::{
    core::{
        algebra::{Matrix4, Point3},
        arrayvec::ArrayVec,
        math::TriangleDefinition,
        parking_lot::Mutex,
        pool::Handle,
        scope_profile,
        sstorage::ImmutableString,
    },
    material::{Material, PropertyValue},
    scene::{
        graph::Graph,
        mesh::{
            buffer::{GeometryBuffer, VertexAttributeUsage, VertexReadTrait, VertexWriteTrait},
            surface::SurfaceData,
            RenderPath,
        },
        node::Node,
    },
    utils::log::{Log, MessageKind},
};
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    fmt::{Debug, Formatter},
    hash::Hasher,
    sync::Arc,
};

pub const BONE_MATRICES_COUNT: usize = 64;

pub struct SurfaceInstance {
    /// Can be [`Handle::NONE`] if it is a fake surface instance for batching.
    pub owner: Handle<Node>,
    pub world_transform: Matrix4<f32>,
    pub bone_matrices: ArrayVec<Matrix4<f32>, BONE_MATRICES_COUNT>,
    pub depth_offset: f32,
}

pub struct Batch {
    pub data: Arc<Mutex<SurfaceData>>,
    pub instances: Vec<SurfaceInstance>,
    pub material: Arc<Mutex<Material>>,
    pub is_skinned: bool,
    pub render_path: RenderPath,
    sort_index: u64,
}

impl Debug for Batch {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Batch {}: {} instances",
            &*self.data as *const _ as u64,
            self.instances.len()
        )
    }
}

#[derive(Default)]
pub struct BatchStorage {
    buffers: Vec<Vec<SurfaceInstance>>,
    batch_map: HashMap<u64, usize>,
    /// Sorted list of batches.
    pub batches: Vec<Batch>,
}

impl BatchStorage {
    /// Puts instances of each batch into a single vertex and index buffers.
    fn try_optimize_batches(&mut self) {
        for batch in self
            .batches
            .iter_mut()
            .filter(|b| b.instances.len() > 1 && !b.is_skinned)
        {
            let src_data = batch.data.lock();

            // Clone vertex buffer and duplicate its contents n times.
            let mut vertex_buffer = src_data.vertex_buffer.clone();
            let mut vertex_buffer_ref_mut = vertex_buffer.modify();
            vertex_buffer_ref_mut.multiplicate(batch.instances.len() as u32);

            let mut triangles = Vec::new();

            let mut iterator = vertex_buffer_ref_mut.iter_mut();
            let mut start_index = 0;
            for instance in batch.instances.iter() {
                // Transform vertices first.
                for _ in 0..src_data.vertex_buffer.vertex_count() {
                    let mut vertex = iterator.next().unwrap();

                    if let Ok(position) = vertex.read_3_f32(VertexAttributeUsage::Position) {
                        vertex
                            .write_3_f32(
                                VertexAttributeUsage::Position,
                                instance
                                    .world_transform
                                    .transform_point(&Point3::from(position))
                                    .coords,
                            )
                            .unwrap();
                    }

                    if let Ok(normal) = vertex.read_3_f32(VertexAttributeUsage::Normal) {
                        vertex
                            .write_3_f32(
                                VertexAttributeUsage::Normal,
                                instance.world_transform.transform_vector(&normal),
                            )
                            .unwrap();
                    }

                    if let Ok(tangent) = vertex.read_3_f32(VertexAttributeUsage::Tangent) {
                        vertex
                            .write_3_f32(
                                VertexAttributeUsage::Tangent,
                                instance.world_transform.transform_vector(&tangent),
                            )
                            .unwrap();
                    }
                }

                // Fill triangles and offset each block.
                for triangle in src_data.geometry_buffer.iter() {
                    triangles.push(TriangleDefinition([
                        triangle[0] + start_index,
                        triangle[1] + start_index,
                        triangle[2] + start_index,
                    ]))
                }
                start_index += src_data.vertex_buffer.vertex_count();
            }

            drop(iterator);
            drop(vertex_buffer_ref_mut);
            drop(src_data);

            // Replace batch data for rendering.
            let new_batch_data =
                SurfaceData::new(vertex_buffer, GeometryBuffer::new(triangles), true);
            batch.data = Arc::new(Mutex::new(new_batch_data));

            // We also need to ensure that new data won't be rendered multiple times.
            batch.instances.clear();
            batch.instances.push(SurfaceInstance {
                owner: Default::default(),
                // Pass identity matrix since our geometry was already transformed.
                world_transform: Matrix4::identity(),
                bone_matrices: Default::default(),
                depth_offset: 0.0,
            })
        }
    }

    pub(in crate) fn generate_batches(&mut self, graph: &Graph) {
        scope_profile!();

        for batch in self.batches.iter_mut() {
            batch.instances.clear();
            self.buffers.push(std::mem::take(&mut batch.instances));
        }

        self.batches.clear();
        self.batch_map.clear();

        for (handle, node) in graph.pair_iter() {
            match node {
                Node::Mesh(mesh) => {
                    for surface in mesh.surfaces().iter() {
                        let is_skinned = !surface.bones.is_empty();

                        let world = if is_skinned {
                            Matrix4::identity()
                        } else {
                            mesh.global_transform()
                        };

                        let data = surface.data();
                        let batch_id = surface.batch_id();

                        let batch = if let Some(&batch_index) = self.batch_map.get(&batch_id) {
                            self.batches.get_mut(batch_index).unwrap()
                        } else {
                            self.batch_map.insert(batch_id, self.batches.len());
                            self.batches.push(Batch {
                                data,
                                // Batches from meshes will be sorted using materials.
                                // This will significantly reduce pipeline state changes.
                                sort_index: surface.material_id(),
                                instances: self.buffers.pop().unwrap_or_default(),
                                material: surface.material().clone(),
                                is_skinned: !surface.bones.is_empty(),
                                render_path: mesh.render_path(),
                            });
                            self.batches.last_mut().unwrap()
                        };

                        batch.sort_index = surface.material_id();
                        batch.material = surface.material().clone();

                        batch.instances.push(SurfaceInstance {
                            world_transform: world,
                            bone_matrices: surface
                                .bones
                                .iter()
                                .map(|&bone_handle| {
                                    let bone_node = &graph[bone_handle];
                                    bone_node.global_transform()
                                        * bone_node.inv_bind_pose_transform()
                                })
                                .collect(),
                            owner: handle,
                            depth_offset: mesh.depth_offset_factor(),
                        });
                    }
                }
                Node::Terrain(terrain) => {
                    for (layer_index, layer) in terrain.layers().iter().enumerate() {
                        for (chunk_index, chunk) in terrain.chunks_ref().iter().enumerate() {
                            let data = chunk.data();
                            let data_key = &*data as *const _ as u64;

                            let mut material = (*layer.material.lock()).clone();
                            match material.set_property(
                                &ImmutableString::new(&layer.mask_property_name),
                                PropertyValue::Sampler {
                                    value: Some(layer.chunk_masks[chunk_index].clone()),
                                    fallback: Default::default(),
                                },
                            ) {
                                Ok(_) => {
                                    let material = Arc::new(Mutex::new(material));

                                    let mut hasher = DefaultHasher::new();

                                    hasher.write_u64(&*material as *const _ as u64);
                                    hasher.write_u64(data_key);

                                    let key = hasher.finish();

                                    let batch = if let Some(&batch_index) = self.batch_map.get(&key)
                                    {
                                        self.batches.get_mut(batch_index).unwrap()
                                    } else {
                                        self.batch_map.insert(key, self.batches.len());
                                        self.batches.push(Batch {
                                            data: data.clone(),
                                            instances: self.buffers.pop().unwrap_or_default(),
                                            material: material.clone(),
                                            is_skinned: false,
                                            render_path: RenderPath::Deferred,
                                            sort_index: layer_index as u64,
                                        });
                                        self.batches.last_mut().unwrap()
                                    };

                                    batch.sort_index = layer_index as u64;
                                    batch.material = material;

                                    batch.instances.push(SurfaceInstance {
                                        world_transform: terrain.global_transform(),
                                        bone_matrices: Default::default(),
                                        owner: handle,
                                        depth_offset: terrain.depth_offset_factor(),
                                    });
                                }
                                Err(e) => Log::writeln(
                                    MessageKind::Error,
                                    format!(
                                        "Failed to prepare batch for terrain chunk.\
                                 Unable to set mask texture for terrain material. Reason: {:?}",
                                        e
                                    ),
                                ),
                            }
                        }
                    }
                }
                _ => (),
            }
        }

        for batch in self.batches.iter_mut() {
            batch.instances.shrink_to_fit();
        }

        self.batches.sort_unstable_by_key(|b| b.sort_index);

        self.try_optimize_batches();
    }
}
