struct Vertex { position: vec4<f32>, normal: vec4<f32> }
struct Params { matrix: mat4x4<f32>, color: vec4<f32>, counts: vec4<u32> }
@group(0) @binding(0) var<storage, read> base: array<Vertex>;
@group(0) @binding(1) var<storage, read> deltas: array<Vertex>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<Vertex>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(64)
fn morph(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.counts.x { return; }
    var p = base[i].position;
    var n = base[i].normal;
    for(var t = 0u; t < params.counts.y; t++) {
        let delta = deltas[t * params.counts.x + i];
        p += weights[t] * delta.position;
        n += weights[t] * delta.normal;
    }
    var normalized = vec3<f32>(0.0, 0.0, 1.0);
    if length(n.xyz) > 1e-8 { normalized = normalize(n.xyz); }
    result[i] = Vertex(vec4<f32>(p.xyz,1.0), vec4<f32>(normalized,0.0));
}
