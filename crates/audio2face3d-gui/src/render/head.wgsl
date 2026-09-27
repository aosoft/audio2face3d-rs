struct Params { matrix: mat4x4<f32>, color: vec4<f32>, counts: vec4<u32> }
@group(0) @binding(0) var<uniform> params: Params;
struct Output { @builtin(position) position: vec4<f32>, @location(0) normal: vec3<f32> }
@vertex fn vertex(@location(0) position: vec4<f32>, @location(1) normal: vec4<f32>) -> Output {
    return Output(params.matrix * position, normal.xyz);
}
@fragment fn fragment(input: Output, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let normal = normalize(input.normal) * select(-1.0, 1.0, front);
    let light = 0.30 + 0.70 * max(dot(normal, normalize(vec3<f32>(-0.4,0.7,1.0))),0.0);
    return vec4<f32>(params.color.rgb * light,1.0);
}
