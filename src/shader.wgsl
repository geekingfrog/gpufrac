// vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct View {
    clip_center: vec2<f32>,
    size_px: vec2<f32>,
    zoom: f32,
}

@group(0) @binding(0)
var<uniform> view: View;

struct VertexOutput {
    // must return @builtin(position)
    @builtin(position) position: vec4<f32>,
    // used to actually compute stuff
    @location(0) clip_position: vec2<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(model.position, 1.0);
    out.clip_position = vec2f(model.position.xy);
    return out;
}


// fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ratio = view.size_px.y / view.size_px.x;
    var c = in.clip_position + view.clip_center;
    c = vec2f(c.x, c.y * ratio) / view.zoom;
    var z = vec2f(0.0);
    let max = 255;
    for (var i = 0; i < max; i++) {
        z = vec2f(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        if z.x * z.x + z.y * z.y > 4.0 {
            return vec4f(f32(i) / f32(max), 0.0, 0.0, 1.0);
        }
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
