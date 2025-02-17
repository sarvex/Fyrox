(
    name: "CustomShader",

    properties: [
        (
            name: "diffuseTexture",
            kind: Sampler(default: None, fallback: White),
        ),
        (
            name: "time",
            kind: Float(0.0),
        )
    ],

    passes: [
        (
            name: "GBuffer",
            draw_parameters: DrawParameters(
                cull_face: Some(Back),
                color_write: ColorMask(
                    red: true,
                    green: true,
                    blue: true,
                    alpha: true,
                ),
                depth_write: true,
                stencil_test: None,
                depth_test: true,
                blend: None,
                stencil_op: StencilOp(
                    fail: Keep,
                    zfail: Keep,
                    zpass: Keep,
                    write_mask: 0xFFFF_FFFF,
                ),
            ),
            vertex_shader:
            r#"
            layout(location = 0) in vec3 vertexPosition;
            layout(location = 1) in vec2 vertexTexCoord;
            layout(location = 2) in vec3 vertexNormal;

            uniform float time;

            // Define uniforms with reserved names. Fyrox will automatically provide
            // required data to these uniforms.
            uniform mat4 fyrox_worldMatrix;
            uniform mat4 fyrox_worldViewProjection;

            out vec3 position;
            out vec3 normal;
            out vec2 texCoord;

            vec3 Wave(vec3 p)
            {
                float x = cos(25.0 * p.y + 30.0 * p.x + 25.0 * p.z + 6.28 * time) * 0.05;
                float y = sin(25.0 * p.y + 20.0 * p.x + 25.0 * p.z + 6.28 * time) * 0.05;
                return vec3(p.x + x, p.y + y, p.z);
            }

            void main()
            {
                texCoord = vertexTexCoord;
                normal = normalize(mat3(fyrox_worldMatrix) * vertexNormal);
                gl_Position = fyrox_worldViewProjection * vec4(Wave(vertexPosition), 1.0);
            }
            "#,

            fragment_shader:
            r#"
            layout(location = 0) out vec4 outColor;
            layout(location = 1) out vec4 outNormal;
            layout(location = 2) out vec4 outAmbient;
            layout(location = 3) out vec4 outMaterial;
            layout(location = 4) out uint outDecalMask;

            // Properties.
            uniform sampler2D diffuseTexture;

            in vec3 normal;
            in vec2 texCoord;

            void main()
            {
                outColor = texture(diffuseTexture, texCoord);
                outNormal = vec4(normal * 0.5 + 0.5, 1.0);
                outMaterial = vec4(0.0, 1.0, 0.0, 1.0);
                outAmbient = vec4(0.0, 0.0, 0.0, 1.0);
                outDecalMask = 0u;
            }
            "#,
        ),

        (
            name: "SpotShadow",

            draw_parameters: DrawParameters (
                cull_face: Some(Back),
                color_write: ColorMask(
                    red: false,
                    green: false,
                    blue: false,
                    alpha: false,
                ),
                depth_write: true,
                stencil_test: None,
                depth_test: true,
                blend: None,
                stencil_op: StencilOp(
                    fail: Keep,
                    zfail: Keep,
                    zpass: Keep,
                    write_mask: 0xFFFF_FFFF,
                ),
            ),

            vertex_shader:
            r#"
            layout(location = 0) in vec3 vertexPosition;
            layout(location = 1) in vec2 vertexTexCoord;

            uniform float time;

            uniform mat4 fyrox_worldViewProjection;

            out vec2 texCoord;

            vec3 Wave(vec3 p)
            {
                float x = cos(25.0 * p.y + 30.0 * p.x + 25.0 * p.z + 6.28 * time) * 0.05;
                float y = sin(25.0 * p.y + 20.0 * p.x + 25.0 * p.z + 6.28 * time) * 0.05;
                return vec3(p.x + x, p.y + y, p.z);
            }

            void main()
            {
                gl_Position = fyrox_worldViewProjection * vec4(Wave(vertexPosition), 1.0);
                texCoord = vertexTexCoord;
            }
            "#,

            fragment_shader:
            r#"
            uniform sampler2D diffuseTexture;

            in vec2 texCoord;

            void main()
            {
                if (texture(diffuseTexture, texCoord).a < 0.2) discard;
            }
            "#,
        ),
        (
            name: "PointShadow",

            draw_parameters: DrawParameters (
                cull_face: Some(Back),
                color_write: ColorMask(
                    red: true,
                    green: true,
                    blue: true,
                    alpha: true,
                ),
                depth_write: true,
                stencil_test: None,
                depth_test: true,
                blend: None,
                stencil_op: StencilOp(
                    fail: Keep,
                    zfail: Keep,
                    zpass: Keep,
                    write_mask: 0xFFFF_FFFF,
                ),
            ),

            vertex_shader:
            r#"
            layout(location = 0) in vec3 vertexPosition;
            layout(location = 1) in vec2 vertexTexCoord;

            uniform mat4 fyrox_worldMatrix;
            uniform mat4 fyrox_worldViewProjection;

            uniform float time;

            out vec2 texCoord;
            out vec3 worldPosition;

            vec3 Wave(vec3 p)
            {
                float x = cos(25.0 * p.y + 30.0 * p.x + 25.0 * p.z + 6.28 * time) * 0.05;
                float y = sin(25.0 * p.y + 20.0 * p.x + 25.0 * p.z + 6.28 * time) * 0.05;
                return vec3(p.x + x, p.y + y, p.z);
            }

            void main()
            {
                gl_Position = fyrox_worldViewProjection * vec4(Wave(vertexPosition), 1.0);
                worldPosition = (fyrox_worldMatrix * vec4(vertexPosition, 1.0)).xyz;
                texCoord = vertexTexCoord;
            }
            "#,

            fragment_shader:
            r#"
            uniform sampler2D diffuseTexture;

            uniform vec3 fyrox_lightPosition;

            in vec2 texCoord;
            in vec3 worldPosition;

            layout(location = 0) out float depth;

            void main()
            {
                if (texture(diffuseTexture, texCoord).a < 0.2) discard;
                depth = length(fyrox_lightPosition - worldPosition);
            }
            "#,
        )
    ],
)