// vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct View {
    clip_center: vec2<f32>,
    clip_width: f32,
    size_px: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> view: View;

@group(1) @binding(0)
var<uniform> julia_constant: vec2f;

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

// how many iteration before assuming the point is in the set
const MAX_I: i32 = 400;

fn juliabrot(z0: vec2<f32>, c: vec2<f32>) -> i32 {
    var z = z0;
    for (var i = 0; i < MAX_I; i++) {
        z = vec2f(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        if z.x * z.x + z.y * z.y > 4.0 {
            return i;
        }
    }
    return MAX_I;
}

fn mandelbrot(pos: vec2<f32>) -> i32 {
    let ratio = view.size_px.y / view.size_px.x;
    var c = pos + view.clip_center;
    c = vec2f(c.x, c.y * ratio) * view.clip_width / 2;
    var z = vec2f(0.0);
    return juliabrot(z, c);
}

fn colorscheme(i: i32) -> vec4f {
    let x = f32(i) / f32(MAX_I) * 100;
    let band_width: f32 = 100 / 8;

    // let ratio = f32(i % i32(band_width)) / band_width;
    let ratio = mix(0, 1, x/100);
    let vx = vec4f(ratio, ratio, ratio, 1);

    if x <= band_width {
        return mix(vec4f(0,0,0,1), vec4f(1,0,0,1), vx);
    }

    if x <= band_width * 2 {
        return mix(vec4f(1,0,0,1), vec4f(1,0.5,0,1), vx);
    }

    if x <= band_width * 3 {
        return mix(vec4f(1,0.5,0,1), vec4f(1,1,0,1), vx);
    }

    if x <= band_width * 4 {
        return mix(vec4f(1,1,0,1), vec4f(0,1,0,1), vx);
    }

    if x <= band_width * 5 {
        return mix(vec4f(0,1,0,1), vec4f(0,0,1,1), vx);
    }

    if x <= band_width * 6 {
        return mix(vec4f(0,0,1,1), vec4f(0.3,0,0.507,1), vx);
    }

    if x <= band_width * 7 {
        return mix(vec4f(0.3,0,0.507,1), vec4f(0.57,0,0.824,1), vx);
    }

    if x <= 0.99 {
        return mix(vec4f(0.57,0,0.824,1), vec4f(1,1,1,1), vx);
    }
    return vec4f(0,0,0,1);
}

// fragment shader

@fragment
fn fs_mandelbrot(in: VertexOutput) -> @location(0) vec4<f32> {
    let ratio = view.size_px.y / view.size_px.x;
    let step = vec2f(1, ratio) / (view.clip_width * view.size_px.x / 2);
    var sum: f32 = 0;
    var c = mandelbrot(in.clip_position);
    return colorscheme(c);
}

@fragment
fn fs_julia(in: VertexOutput) -> @location(0) vec4<f32> {
    let ratio = view.size_px.y / view.size_px.x;
    var c = in.clip_position + view.clip_center;
    var z = vec2f(c.x, c.y * ratio) * view.clip_width / 2;
    let i = juliabrot(z, julia_constant);

    return colorscheme(i);
}
