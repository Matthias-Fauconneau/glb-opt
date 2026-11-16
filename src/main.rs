use std::{borrow::Cow, fs, path::PathBuf};

use anyhow::{Context, ensure};
use clap::Parser;
use gltf::json::Value;
use meshoptimizer_rs::{
    Attributes, Positions, SimplifyOptions, SimplifySettings, VertexStream, Workspace,
    optimize_vertex_cache, optimize_vertex_fetch, simplify, simplify_with_attributes,
};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Simplify triangle meshes in a binary glTF (GLB) file"
)]
struct Args {
    /// Input .glb file
    input: PathBuf,
    /// Output .glb file
    output: PathBuf,
    /// Fraction of triangles to keep, in the range (0, 1]
    #[arg(short, long, default_value_t = 0.5, value_parser = parse_ratio)]
    ratio: f32,
    /// Maximum normalized geometric error allowed by meshoptimizer
    #[arg(long, default_value_t = 0.01)]
    error: f32,
    /// Include COLOR_0 values in the simplification error metric when present
    #[arg(long)]
    preserve_colors: bool,
}

fn parse_ratio(value: &str) -> Result<f32, String> {
    let ratio: f32 = value
        .parse()
        .map_err(|_| "ratio must be a number".to_owned())?;
    if ratio.is_finite() && ratio > 0.0 && ratio <= 1.0 {
        Ok(ratio)
    } else {
        Err("ratio must be greater than 0 and at most 1".to_owned())
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let bytes =
        fs::read(&args.input).with_context(|| format!("reading {}", args.input.display()))?;
    let output = simplify_glb(&bytes, args.ratio, args.error, args.preserve_colors)?;
    fs::write(&args.output, output)
        .with_context(|| format!("writing {}", args.output.display()))?;
    Ok(())
}

fn simplify_glb(
    bytes: &[u8],
    ratio: f32,
    error: f32,
    preserve_colors: bool,
) -> anyhow::Result<Vec<u8>> {
    ensure!(
        ratio.is_finite() && ratio > 0.0 && ratio <= 1.0,
        "ratio must be greater than 0 and at most 1"
    );
    ensure!(
        error.is_finite() && error >= 0.0,
        "--error must be finite and non-negative"
    );
    let mut glb = gltf::Glb::from_slice(&bytes).context("parsing GLB container")?;
    let mut json_chunk = glb.json.to_vec();
    let mut bin_chunk = glb.bin.take().context("GLB has no BIN chunk")?.into_owned();
    let source_bin = bin_chunk.clone();
    let (document, buffers, _) = gltf::import_slice(&bytes).context("decoding GLB")?;
    ensure!(buffers.len() == 1, "GLB must contain exactly one buffer");

    let mut root: Value =
        gltf::json::deserialize::from_slice(&json_chunk).context("parsing GLB JSON")?;
    ensure!(
        root["buffers"].as_array().is_some_and(|b| b.len() == 1),
        "expected one GLB buffer"
    );
    ensure!(
        root["buffers"][0].get("uri").is_none(),
        "GLB buffer must be embedded in the BIN chunk"
    );
    let mut accessor_use_counts = count_accessor_uses(&root);
    let has_extensions = contains_key(&root, "extensions");

    let mut workspace = Workspace::default();

    for mesh in document.meshes() {
        for (primitive_index, primitive) in mesh.primitives().enumerate() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = primitive
                .reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
            let Some(positions) = reader.read_positions() else {
                continue;
            };
            let positions: Vec<[f32; 3]> = positions.collect();
            if positions.is_empty() {
                continue;
            }
            let indices: Vec<u32> = reader
                .read_indices()
                .map(|i| i.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());
            ensure!(
                indices.len() % 3 == 0,
                "mesh {} primitive {} has a non-triangle index count",
                mesh.index(),
                primitive_index
            );
            if indices.is_empty() {
                continue;
            }

            let target_index_count =
                (((indices.len() as f32 * ratio).floor() as usize).max(3) / 3) * 3;
            let settings = SimplifySettings {
                target_index_count,
                target_error: error,
                options: SimplifyOptions::EMPTY,
            };
            let simplified = if preserve_colors {
                if let Some(colors) = reader.read_colors(0) {
                    let colors: Vec<f32> = colors
                        .into_rgba_f32()
                        .flat_map(|rgba| rgba.into_iter())
                        .collect();
                    ensure!(
                        colors.len() == positions.len() * 4,
                        "mesh {} primitive {} has a COLOR_0 count that does not match POSITION",
                        mesh.index(),
                        primitive_index
                    );
                    let attributes =
                        Attributes::from_interleaved(&colors, positions.len(), 4, 4, 0)
                            .context("building vertex color attributes")?;
                    simplify_with_attributes(
                        &indices,
                        Positions::from_packed(&positions),
                        attributes,
                        &[1.0; 4],
                        None,
                        settings,
                        &mut workspace,
                    )?
                    .indices
                } else {
                    simplify(
                        &indices,
                        Positions::from_packed(&positions),
                        settings,
                        &mut workspace,
                    )?
                    .indices
                }
            } else {
                simplify(
                    &indices,
                    Positions::from_packed(&positions),
                    settings,
                    &mut workspace,
                )?
                .indices
            }
            .to_vec();
            let cache_optimized =
                optimize_vertex_cache(&simplified, positions.len(), &mut workspace)?;
            let position_bytes: Vec<u8> = positions
                .iter()
                .flat_map(|position| {
                    position
                        .iter()
                        .flat_map(|component| component.to_le_bytes())
                })
                .collect();
            let fetched = optimize_vertex_fetch(
                &cache_optimized,
                VertexStream::packed(&position_bytes, 12)?,
                &mut workspace,
            )?;
            let mut new_to_old = vec![u32::MAX; fetched.vertex_count];
            for (&old_index, &new_index) in cache_optimized.iter().zip(&fetched.indices) {
                let slot = new_to_old
                    .get_mut(new_index as usize)
                    .context("fetch optimizer returned an invalid vertex index")?;
                ensure!(
                    *slot == u32::MAX || *slot == old_index,
                    "fetch optimizer returned an inconsistent vertex remap"
                );
                *slot = old_index;
            }
            ensure!(
                new_to_old.iter().all(|&index| index != u32::MAX),
                "fetch optimizer left an unmapped output vertex"
            );

            compact_primitive_attributes(
                &mut root,
                mesh.index(),
                primitive_index,
                &source_bin,
                &new_to_old,
                &fetched.vertices,
                positions.len(),
                &mut accessor_use_counts,
                has_extensions,
                &mut bin_chunk,
            )?;
            let simplified = fetched.indices.as_slice();

            let (component_type, encoded) =
                if simplified.iter().all(|&index| index <= u16::MAX as u32) {
                    (
                        5123,
                        simplified
                            .iter()
                            .flat_map(|&index| (index as u16).to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                } else {
                    (
                        5125,
                        simplified
                            .iter()
                            .flat_map(|index| index.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                };
            while bin_chunk.len() % 4 != 0 {
                bin_chunk.push(0);
            }
            let byte_offset = bin_chunk.len();
            bin_chunk.extend_from_slice(&encoded);
            let old_index_accessor =
                root["meshes"][mesh.index()]["primitives"][primitive_index]["indices"]
                    .as_u64()
                    .map(|index| index as usize);
            let accessor_index = append_index_accessor(
                &mut root,
                old_index_accessor,
                &mut accessor_use_counts,
                has_extensions,
                component_type,
                simplified,
                encoded,
                byte_offset,
            )?;
            root["meshes"][mesh.index()]["primitives"][primitive_index]["indices"] =
                json!(accessor_index);
        }
    }

    bin_chunk = repack_buffer_views(&mut root, &bin_chunk)?;
    ensure!(
        root["buffers"][0]["byteLength"].as_u64().is_some(),
        "buffer has no byteLength"
    );
    root["buffers"][0]["byteLength"] = json!(bin_chunk.len());
    json_chunk = gltf::json::serialize::to_vec(&root).context("serializing GLB JSON")?;
    glb.json = Cow::Owned(json_chunk);
    glb.bin = Some(Cow::Owned(bin_chunk));
    let output = glb.to_vec().context("serializing GLB container")?;
    Ok(output)
}

fn compact_primitive_attributes(
    root: &mut Value,
    mesh_index: usize,
    primitive_index: usize,
    source_bin: &[u8],
    new_to_old: &[u32],
    fetched_positions: &[u8],
    source_vertex_count: usize,
    remaining_accessor_uses: &mut [usize],
    has_extensions: bool,
    output_bin: &mut Vec<u8>,
) -> anyhow::Result<()> {
    let primitive = &root["meshes"][mesh_index]["primitives"][primitive_index];
    let position_accessor = primitive["attributes"]["POSITION"]
        .as_u64()
        .context("primitive has no POSITION accessor")? as usize;
    let mut accessor_ids = Vec::new();
    let attributes = primitive["attributes"]
        .as_object()
        .context("primitive attributes are not an object")?;
    for accessor in attributes.values() {
        accessor_ids.push(
            accessor
                .as_u64()
                .context("primitive attribute accessor is not an index")? as usize,
        );
    }
    if let Some(targets) = primitive["targets"].as_array() {
        for target in targets {
            for accessor in target
                .as_object()
                .context("morph target attributes are not an object")?
                .values()
            {
                accessor_ids.push(
                    accessor
                        .as_u64()
                        .context("morph target accessor is not an index")?
                        as usize,
                );
            }
        }
    }
    accessor_ids.sort_unstable();
    accessor_ids.dedup();

    let mut local_uses = std::collections::HashMap::<usize, usize>::new();
    for accessor in attributes.values() {
        *local_uses
            .entry(accessor.as_u64().unwrap() as usize)
            .or_default() += 1;
    }
    if let Some(targets) = primitive["targets"].as_array() {
        for target in targets {
            for accessor in target.as_object().unwrap().values() {
                *local_uses
                    .entry(accessor.as_u64().unwrap() as usize)
                    .or_default() += 1;
            }
        }
    }

    let mut replacements = std::collections::HashMap::new();
    for accessor_index in accessor_ids {
        let data = if accessor_index == position_accessor {
            ensure!(
                fetched_positions.len() == new_to_old.len() * 12,
                "fetch optimizer returned an invalid position stream"
            );
            fetched_positions.to_vec()
        } else {
            compact_accessor_data(
                root,
                accessor_index,
                source_bin,
                new_to_old,
                source_vertex_count,
            )?
        };
        let use_count = local_uses[&accessor_index];
        let can_reuse = !has_extensions
            && remaining_accessor_uses.get(accessor_index).copied() == Some(use_count);
        let replacement = append_compacted_accessor(
            root,
            accessor_index,
            data,
            new_to_old.len(),
            accessor_index == position_accessor,
            can_reuse,
            output_bin,
        )?;
        if let Some(remaining) = remaining_accessor_uses.get_mut(accessor_index) {
            *remaining = remaining.saturating_sub(use_count);
        }
        replacements.insert(accessor_index, replacement);
    }

    let primitive = &mut root["meshes"][mesh_index]["primitives"][primitive_index];
    if let Some(attributes) = primitive["attributes"].as_object_mut() {
        for accessor in attributes.values_mut() {
            let old_index = accessor.as_u64().context("invalid attribute accessor")? as usize;
            *accessor = json!(replacements[&old_index]);
        }
    }
    if let Some(targets) = primitive["targets"].as_array_mut() {
        for target in targets {
            if let Some(attributes) = target.as_object_mut() {
                for accessor in attributes.values_mut() {
                    let old_index =
                        accessor.as_u64().context("invalid morph target accessor")? as usize;
                    *accessor = json!(replacements[&old_index]);
                }
            }
        }
    }
    Ok(())
}

fn compact_accessor_data(
    root: &Value,
    accessor_index: usize,
    source_bin: &[u8],
    new_to_old: &[u32],
    source_vertex_count: usize,
) -> anyhow::Result<Vec<u8>> {
    let accessor = &root["accessors"][accessor_index];
    ensure!(
        accessor.is_object(),
        "vertex accessor {accessor_index} is missing"
    );
    ensure!(
        accessor.get("sparse").is_none(),
        "cannot compact sparse vertex accessor {accessor_index}"
    );
    ensure!(
        accessor["count"].as_u64() == Some(source_vertex_count as u64),
        "vertex accessor {accessor_index} count does not match POSITION"
    );
    let view_index = accessor["bufferView"]
        .as_u64()
        .context("vertex accessor has no bufferView")? as usize;
    let view = &root["bufferViews"][view_index];
    ensure!(
        view["buffer"].as_u64() == Some(0),
        "vertex accessor references a non-GLB buffer"
    );
    let component_size = match accessor["componentType"].as_u64() {
        Some(5120 | 5121) => 1usize,
        Some(5122 | 5123) => 2usize,
        Some(5125 | 5126) => 4usize,
        _ => anyhow::bail!("unsupported vertex accessor component type"),
    };
    let component_count = match accessor["type"].as_str() {
        Some("SCALAR") => 1usize,
        Some("VEC2") => 2,
        Some("VEC3") => 3,
        Some("VEC4") => 4,
        _ => anyhow::bail!("unsupported vertex accessor type"),
    };
    let element_size = component_size * component_count;
    let stride = view["byteStride"].as_u64().unwrap_or(element_size as u64) as usize;
    ensure!(
        stride >= element_size,
        "vertex bufferView stride is too small"
    );
    let base_offset = view["byteOffset"].as_u64().unwrap_or(0) as usize
        + accessor["byteOffset"].as_u64().unwrap_or(0) as usize;
    let mut compacted = Vec::with_capacity(new_to_old.len() * element_size);
    for &old_index in new_to_old {
        ensure!(
            (old_index as usize) < source_vertex_count,
            "invalid old vertex index"
        );
        let start = base_offset
            .checked_add(old_index as usize * stride)
            .context("vertex accessor byte offset overflow")?;
        let end = start
            .checked_add(element_size)
            .context("vertex accessor byte range overflow")?;
        let element = source_bin
            .get(start..end)
            .context("vertex accessor exceeds the GLB BIN chunk")?;
        compacted.extend_from_slice(element);
    }
    Ok(compacted)
}

fn append_compacted_accessor(
    root: &mut Value,
    source_accessor_index: usize,
    data: Vec<u8>,
    vertex_count: usize,
    is_position: bool,
    reuse_source: bool,
    output_bin: &mut Vec<u8>,
) -> anyhow::Result<usize> {
    while output_bin.len() % 4 != 0 {
        output_bin.push(0);
    }
    let byte_offset = output_bin.len();
    let byte_length = data.len();
    output_bin.extend_from_slice(&data);

    let view_index = root["bufferViews"]
        .as_array()
        .map_or(0, |views| views.len());
    root["bufferViews"]
        .as_array_mut()
        .context("missing bufferViews array")?
        .push(json!({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": byte_length,
            "target": 34962
        }));

    let mut accessor = root["accessors"][source_accessor_index].clone();
    accessor["bufferView"] = json!(view_index);
    accessor["byteOffset"] = json!(0);
    accessor["count"] = json!(vertex_count);
    accessor
        .as_object_mut()
        .context("accessor is not an object")?
        .remove("min");
    accessor
        .as_object_mut()
        .context("accessor is not an object")?
        .remove("max");
    accessor
        .as_object_mut()
        .context("accessor is not an object")?
        .remove("sparse");
    if is_position {
        ensure!(
            data.len() == vertex_count * 12,
            "compacted POSITION data has an invalid size"
        );
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for vertex in data.chunks_exact(12) {
            for component in 0..3 {
                let offset = component * 4;
                let value = f32::from_le_bytes(vertex[offset..offset + 4].try_into()?);
                min[component] = min[component].min(value);
                max[component] = max[component].max(value);
            }
        }
        accessor["min"] = json!(min);
        accessor["max"] = json!(max);
    }

    let accessor_index = if reuse_source {
        root["accessors"][source_accessor_index] = accessor;
        source_accessor_index
    } else {
        let index = root["accessors"]
            .as_array()
            .map_or(0, |accessors| accessors.len());
        root["accessors"]
            .as_array_mut()
            .context("missing accessors array")?
            .push(accessor);
        index
    };
    Ok(accessor_index)
}

fn append_index_accessor(
    root: &mut Value,
    source_accessor_index: Option<usize>,
    remaining_accessor_uses: &mut [usize],
    has_extensions: bool,
    component_type: u32,
    indices: &[u32],
    encoded: Vec<u8>,
    byte_offset: usize,
) -> anyhow::Result<usize> {
    let view_index = root["bufferViews"]
        .as_array()
        .map_or(0, |views| views.len());
    root["bufferViews"]
        .as_array_mut()
        .context("missing bufferViews array")?
        .push(json!({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": encoded.len(),
            "target": 34963
        }));
    let accessor = json!({
        "bufferView": view_index,
        "byteOffset": 0,
        "componentType": component_type,
        "count": indices.len(),
        "type": "SCALAR",
        "min": [indices.iter().copied().min().unwrap_or(0)],
        "max": [indices.iter().copied().max().unwrap_or(0)]
    });
    let reuse_source = source_accessor_index.is_some_and(|index| {
        !has_extensions && remaining_accessor_uses.get(index).copied() == Some(1)
    });
    if let Some(source_index) = source_accessor_index {
        if let Some(remaining) = remaining_accessor_uses.get_mut(source_index) {
            *remaining = remaining.saturating_sub(1);
        }
    }
    if let Some(source_index) = source_accessor_index.filter(|_| reuse_source) {
        root["accessors"][source_index] = accessor;
        Ok(source_index)
    } else {
        let index = root["accessors"]
            .as_array()
            .map_or(0, |accessors| accessors.len());
        root["accessors"]
            .as_array_mut()
            .context("missing accessors array")?
            .push(accessor);
        Ok(index)
    }
}

fn count_accessor_uses(root: &Value) -> Vec<usize> {
    let mut counts = vec![0usize; root["accessors"].as_array().map_or(0, Vec::len)];
    let mut add = |value: &Value| {
        if let Some(index) = value.as_u64().map(|index| index as usize)
            && let Some(count) = counts.get_mut(index)
        {
            *count += 1;
        }
    };
    if let Some(meshes) = root["meshes"].as_array() {
        for mesh in meshes {
            if let Some(primitives) = mesh["primitives"].as_array() {
                for primitive in primitives {
                    if let Some(index) = primitive.get("indices") {
                        if !index.is_null() {
                            add(index);
                        }
                    }
                    if let Some(attributes) = primitive["attributes"].as_object() {
                        for accessor in attributes.values() {
                            add(accessor);
                        }
                    }
                    if let Some(targets) = primitive["targets"].as_array() {
                        for target in targets {
                            if let Some(attributes) = target.as_object() {
                                for accessor in attributes.values() {
                                    add(accessor);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if let Some(animations) = root["animations"].as_array() {
        for animation in animations {
            if let Some(samplers) = animation["samplers"].as_array() {
                for sampler in samplers {
                    add(&sampler["input"]);
                    add(&sampler["output"]);
                }
            }
        }
    }
    counts
}

fn contains_key(value: &Value, key: &str) -> bool {
    match value {
        Value::Object(object) => {
            object.contains_key(key) || object.values().any(|value| contains_key(value, key))
        }
        Value::Array(values) => values.iter().any(|value| contains_key(value, key)),
        _ => false,
    }
}

fn collect_buffer_view_references(
    value: &Value,
    references: &mut std::collections::BTreeSet<usize>,
) {
    match value {
        Value::Object(object) => {
            if let Some(index) = object.get("bufferView").and_then(Value::as_u64) {
                references.insert(index as usize);
            }
            for value in object.values() {
                collect_buffer_view_references(value, references);
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_buffer_view_references(value, references);
            }
        }
        _ => {}
    }
}

fn remap_buffer_view_references(
    value: &mut Value,
    remap: &std::collections::HashMap<usize, usize>,
) -> anyhow::Result<()> {
    match value {
        Value::Object(object) => {
            if let Some(index) = object.get("bufferView").and_then(Value::as_u64) {
                let old_index = index as usize;
                let new_index = remap
                    .get(&old_index)
                    .with_context(|| format!("bufferView {old_index} is not retained"))?;
                object.insert("bufferView".to_owned(), json!(new_index));
            }
            for value in object.values_mut() {
                remap_buffer_view_references(value, remap)?;
            }
        }
        Value::Array(values) => {
            for value in values {
                remap_buffer_view_references(value, remap)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn repack_buffer_views(root: &mut Value, source_bin: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut references = std::collections::BTreeSet::new();
    collect_buffer_view_references(root, &mut references);
    let original_views = root["bufferViews"]
        .as_array()
        .context("missing bufferViews array")?
        .clone();
    let mut remap = std::collections::HashMap::new();
    let mut packed_views = Vec::with_capacity(references.len());
    let mut packed_bin = Vec::new();
    for old_index in references {
        let mut view = original_views
            .get(old_index)
            .with_context(|| format!("bufferView {old_index} is out of range"))?
            .clone();
        ensure!(
            view["buffer"].as_u64() == Some(0),
            "GLB contains a view for a non-embedded buffer"
        );
        while packed_bin.len() % 4 != 0 {
            packed_bin.push(0);
        }
        let offset = view["byteOffset"].as_u64().unwrap_or(0) as usize;
        let length = view["byteLength"]
            .as_u64()
            .context("bufferView has no byteLength")? as usize;
        let end = offset
            .checked_add(length)
            .context("bufferView range overflow")?;
        let bytes = source_bin
            .get(offset..end)
            .with_context(|| format!("bufferView {old_index} exceeds the BIN chunk"))?;
        let new_index = packed_views.len();
        remap.insert(old_index, new_index);
        view["buffer"] = json!(0);
        view["byteOffset"] = json!(packed_bin.len());
        packed_views.push(view);
        packed_bin.extend_from_slice(bytes);
    }
    root["bufferViews"] = json!(packed_views);
    remap_buffer_view_references(root, &remap)?;
    Ok(packed_bin)
}

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, path::Path};

    use bevy::{
        app::{App, TaskPoolPlugin},
        asset::{
            AssetApp, AssetPlugin, AssetServer, Assets, LoadState,
            io::{
                AssetSourceBuilder, AssetSourceId,
                memory::{Dir, MemoryAssetReader},
            },
        },
        gltf::{Gltf, GltfPlugin},
        world_serialization::WorldSerializationPlugin,
    };
    use bevy_log::LogPlugin;
    use bevy_mesh::MeshPlugin;
    use serde_json::json;

    use super::simplify_glb;

    fn synthetic_glb() -> Vec<u8> {
        let positions = [
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let mut bin = Vec::new();
        for position in positions {
            for component in position {
                bin.extend_from_slice(&component.to_le_bytes());
            }
        }
        for index in [0u16, 1, 2, 2, 1, 3] {
            bin.extend_from_slice(&index.to_le_bytes());
        }

        let json = json!({
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{
                "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1 }]
            }],
            "buffers": [{ "byteLength": bin.len() }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 48, "target": 34962 },
                { "buffer": 0, "byteOffset": 48, "byteLength": 12, "target": 34963 }
            ],
            "accessors": [
                {
                    "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                    "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]
                },
                {
                    "bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR",
                    "min": [0], "max": [3]
                }
            ]
        });
        let json = gltf::json::serialize::to_vec(&json).unwrap();
        gltf::Glb {
            header: gltf::binary::Header {
                magic: *b"glTF",
                version: 2,
                length: 0,
            },
            json: Cow::Owned(json),
            bin: Some(Cow::Owned(bin)),
        }
        .to_vec()
        .unwrap()
    }

    #[test]
    fn simplified_glb_loads_with_bevy_gltf_loader() {
        let input = synthetic_glb();
        let output = simplify_glb(&input, 0.5, 1.0, false).unwrap();
        let (document, _, _) = gltf::import_slice(&output).unwrap();
        let primitive = document
            .meshes()
            .next()
            .unwrap()
            .primitives()
            .next()
            .unwrap();
        let positions = primitive.get(&gltf::Semantic::Positions).unwrap();
        assert!(positions.count() <= 4);

        let directory = Dir::default();
        directory.insert_asset(Path::new("simplified.glb"), output);
        let reader = MemoryAssetReader { root: directory };
        let mut app = App::new();
        app.register_asset_source(
            AssetSourceId::Default,
            AssetSourceBuilder::new(move || Box::new(reader.clone())),
        )
        .add_plugins((
            LogPlugin::default(),
            TaskPoolPlugin::default(),
            AssetPlugin::default(),
            WorldSerializationPlugin,
            MeshPlugin,
            GltfPlugin::default(),
        ));
        app.finish();
        app.cleanup();
        app.update();

        // AssetServer dispatches this GLB through bevy_gltf::loader::GltfLoader.
        let asset_server = app.world().resource::<AssetServer>().clone();
        let handle = asset_server.load::<Gltf>("simplified.glb");
        let handle_id = handle.id();
        let mut loaded = false;
        for _ in 0..10_000 {
            app.update();
            match asset_server.get_load_state(handle_id).unwrap() {
                LoadState::Loaded => {
                    loaded = true;
                    break;
                }
                LoadState::Failed(error) => panic!("Bevy failed to load simplified GLB: {error}"),
                _ => {}
            }
        }
        assert!(loaded, "Bevy glTF load did not complete");
        let gltf = app.world().resource::<Assets<Gltf>>().get(&handle).unwrap();
        assert_eq!(gltf.meshes.len(), 1);
    }
}
