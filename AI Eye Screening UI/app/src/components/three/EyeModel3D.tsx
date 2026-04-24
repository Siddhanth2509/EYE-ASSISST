import { useRef, useState, useMemo, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html, Environment, ContactShadows } from '@react-three/drei';
import * as THREE from 'three';

// ============================================================================
// IRIS SHADER - Realistic procedural iris with fractal fibers
// ============================================================================
function IrisMaterial() {
  const materialRef = useRef<THREE.ShaderMaterial>(null);

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uBaseColor: { value: new THREE.Color('#156e83') },
      uInnerColor: { value: new THREE.Color('#0a3d4d') },
      uOuterColor: { value: new THREE.Color('#22D3EE') },
      uSpeckleColor: { value: new THREE.Color('#164E63') },
    }),
    []
  );

  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime * 0.15;
    }
  });

  return (
    <shaderMaterial
      ref={materialRef}
      uniforms={uniforms}
      vertexShader={`
        varying vec2 vUv;
        varying vec3 vPosition;
        void main() {
          vUv = uv;
          vPosition = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `}
      fragmentShader={`
        uniform float uTime;
        uniform vec3 uBaseColor;
        uniform vec3 uInnerColor;
        uniform vec3 uOuterColor;
        uniform vec3 uSpeckleColor;
        varying vec2 vUv;
        varying vec3 vPosition;

        // Hash for noise
        float hash(vec2 p) {
          return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
        }

        // Smooth noise
        float noise(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          float a = hash(i);
          float b = hash(i + vec2(1.0, 0.0));
          float c = hash(i + vec2(0.0, 1.0));
          float d = hash(i + vec2(1.0, 1.0));
          return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }

        // FBM for organic patterns
        float fbm(vec2 p) {
          float value = 0.0;
          float amplitude = 0.5;
          for(int i = 0; i < 5; i++) {
            value += amplitude * noise(p);
            p *= 2.0;
            amplitude *= 0.5;
          }
          return value;
        }

        void main() {
          vec2 center = vUv - 0.5;
          float dist = length(center);
          float angle = atan(center.y, center.x);

          // Outer ring fade
          if (dist > 0.48) discard;
          if (dist < 0.15) discard; // pupil hole

          // Radial iris fibers - main feature
          float radialFibers = 0.0;
          for (float i = 0.0; i < 8.0; i++) {
            float freq = 12.0 + i * 8.0;
            float phase = i * 0.7 + uTime * 0.02;
            radialFibers += sin(angle * freq + phase + fbm(center * 3.0) * 2.0) * (0.5 - i * 0.05);
          }
          radialFibers = radialFibers * 0.5 + 0.5;

          // Collarette ring (inner iris border)
          float collarette = smoothstep(0.16, 0.22, dist) * smoothstep(0.28, 0.22, dist);

          // Color gradient from inner to outer
          vec3 color = mix(uInnerColor, uBaseColor, smoothstep(0.15, 0.35, dist));
          color = mix(color, uOuterColor, smoothstep(0.30, 0.48, dist));

          // Apply fiber texture
          color = mix(color, color * 1.3, radialFibers * 0.4);

          // Collarette detail
          color = mix(color, uSpeckleColor, collarette * 0.6);

          // Crypts (small dark spots near collarette)
          float crypts = fbm(center * 15.0 + vec2(uTime * 0.01));
          crypts = smoothstep(0.4, 0.6, crypts) * collarette;
          color = mix(color, uInnerColor * 0.7, crypts * 0.5);

          // Speckles in outer iris
          float speckles = noise(center * 40.0 + uTime * 0.005);
          speckles = smoothstep(0.55, 0.7, speckles) * smoothstep(0.30, 0.48, dist);
          color = mix(color, uSpeckleColor, speckles * 0.3);

          // Alpha fade at edges
          float alpha = smoothstep(0.48, 0.42, dist) * smoothstep(0.15, 0.18, dist);

          // Slight color variation across the iris
          color += (fbm(center * 8.0) - 0.5) * 0.1;

          gl_FragColor = vec4(color, alpha);
        }
      `}
      transparent
      side={THREE.DoubleSide}
    />
  );
}

// ============================================================================
// SCLERA MATERIAL - Realistic white with subtle veins
// ============================================================================
function ScleraMaterial() {
  return (
    <meshPhysicalMaterial
      color="#F0EDE5"
      roughness={0.45}
      metalness={0.0}
      clearcoat={0.15}
      clearcoatRoughness={0.3}
      sheen={0.2}
      sheenRoughness={0.5}
      sheenColor="#FFE4E1"
    />
  );
}

// ============================================================================
// CORNEA MATERIAL - Ultra-clear with refraction
// ============================================================================
function CorneaMaterial() {
  return (
    <meshPhysicalMaterial
      color="#FFFFFF"
      transparent
      opacity={0.08}
      roughness={0.0}
      metalness={0.0}
      transmission={0.98}
      thickness={0.55}
      ior={1.376}
      clearcoat={1.0}
      clearcoatRoughness={0.0}
    />
  );
}

// ============================================================================
// PUPIL - Realistic with slight edge softness
// ============================================================================
function Pupil({ isExploded }: { isExploded: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (!meshRef.current) return;
    const baseScale = 1.0;
    const breath = 1 + Math.sin(state.clock.elapsedTime * 1.2) * 0.03;
    meshRef.current.scale.setScalar(baseScale * breath);
  });

  return (
    <mesh ref={meshRef} position={[0, 0, isExploded ? 0.75 : 0.43]}>
      <circleGeometry args={[0.16, 64]} />
      <meshBasicMaterial color="#050508" />
    </mesh>
  );
}

// ============================================================================
// LENS - Semi-transparent crystalline
// ============================================================================
function Lens({ isExploded }: { isExploded: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const targetZ = useRef(0.15);

  useFrame((_state, delta) => {
    if (!meshRef.current) return;
    const target = isExploded ? 0.1 : 0.15;
    targetZ.current = THREE.MathUtils.lerp(targetZ.current, target, delta * 4);
    meshRef.current.position.z = targetZ.current;
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.42, 32, 32]} />
      <meshPhysicalMaterial
        color="#FFF8DC"
        transparent
        opacity={0.5}
        roughness={0.1}
        metalness={0.0}
        transmission={0.7}
        thickness={0.4}
        ior={1.41}
      />
    </mesh>
  );
}

// ============================================================================
// RETINA - Detailed with blood vessel pattern
// ============================================================================
function Retina({ isExploded, isHovered }: { isExploded: boolean; isHovered: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const targetZ = useRef(-0.65);

  useFrame((_state, delta) => {
    if (!meshRef.current) return;
    const target = isExploded ? -1.2 : -0.65;
    targetZ.current = THREE.MathUtils.lerp(targetZ.current, target, delta * 4);
    meshRef.current.position.z = targetZ.current;
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.92, 64, 64, 0, Math.PI * 2, Math.PI * 0.55, Math.PI * 0.45]} />
      <meshStandardMaterial
        color={isHovered ? '#FB7185' : '#FFA07A'}
        roughness={0.7}
        metalness={0.05}
        emissive={isHovered ? '#FB7185' : '#1a0505'}
        emissiveIntensity={isHovered ? 0.3 : 0.1}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

// ============================================================================
// OPTIC NERVE
// ============================================================================
function OpticNerve({ isExploded, isHovered }: { isExploded: boolean; isHovered: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const targetZ = useRef(-1.15);

  useFrame((_state, delta) => {
    if (!meshRef.current) return;
    const target = isExploded ? -2.0 : -1.15;
    targetZ.current = THREE.MathUtils.lerp(targetZ.current, target, delta * 4);
    meshRef.current.position.z = targetZ.current;
  });

  return (
    <mesh ref={meshRef}>
      <cylinderGeometry args={[0.10, 0.14, 0.35, 32]} />
      <meshStandardMaterial
        color={isHovered ? '#FCD34D' : '#FFE4B5'}
        roughness={0.65}
        emissive={isHovered ? '#FCD34D' : '#000000'}
        emissiveIntensity={isHovered ? 0.3 : 0}
      />
    </mesh>
  );
}

// ============================================================================
// BLOOD VESSELS - Realistic branching pattern
// ============================================================================
function BloodVessels() {
  const groupRef = useRef<THREE.Group>(null);

  const vesselCurves = useMemo(() => {
    const curves: THREE.CatmullRomCurve3[] = [];
    const seedPoints: { angle: number; branches: number }[] = [
      { angle: 0, branches: 3 },
      { angle: Math.PI * 0.35, branches: 2 },
      { angle: Math.PI * 0.7, branches: 3 },
      { angle: Math.PI, branches: 2 },
      { angle: Math.PI * 1.3, branches: 3 },
      { angle: Math.PI * 1.65, branches: 2 },
    ];

    for (const { angle: baseAngle, branches } of seedPoints) {
      for (let b = 0; b < branches; b++) {
        const points: THREE.Vector3[] = [];
        const branchAngle = baseAngle + (b - branches / 2) * 0.15;
        const numPoints = 6 + Math.floor(Math.random() * 4);

        for (let j = 0; j < numPoints; j++) {
          const t = j / (numPoints - 1);
          const radius = 0.15 + t * 0.7;
          const wobble = Math.sin(t * Math.PI * 3) * 0.08;
          const x = Math.cos(branchAngle + wobble) * radius;
          const y = Math.sin(branchAngle + wobble) * radius;
          const z = Math.sqrt(Math.max(0, 1 - x * x - y * y)) * 0.985;
          points.push(new THREE.Vector3(x, y, z));
        }
        curves.push(new THREE.CatmullRomCurve3(points));
      }
    }
    return curves;
  }, []);

  return (
    <group ref={groupRef}>
      {vesselCurves.map((curve, i) => {
        const thickness = 0.006 + Math.random() * 0.005;
        const opacity = 0.3 + Math.random() * 0.25;
        return (
          <mesh key={i}>
            <tubeGeometry args={[curve, 24, thickness, 6, false]} />
            <meshStandardMaterial
              color="#B22234"
              transparent
              opacity={opacity}
              roughness={0.8}
            />
          </mesh>
        );
      })}
    </group>
  );
}

// ============================================================================
// SCLERA SPHERE
// ============================================================================
function Sclera({ isExploded }: { isExploded: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const targetZ = useRef(0);

  useFrame((_state, delta) => {
    if (!meshRef.current) return;
    const target = isExploded ? -0.3 : 0;
    targetZ.current = THREE.MathUtils.lerp(targetZ.current, target, delta * 4);
    meshRef.current.position.z = targetZ.current;
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[1, 64, 64]} />
      <ScleraMaterial />
    </mesh>
  );
}

// ============================================================================
// IRIS MESH
// ============================================================================
function Iris({ isExploded }: { isExploded: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const targetZ = useRef(0.42);

  useFrame((_state, delta) => {
    if (!meshRef.current) return;
    const target = isExploded ? 0.7 : 0.42;
    targetZ.current = THREE.MathUtils.lerp(targetZ.current, target, delta * 4);
    meshRef.current.position.z = targetZ.current;
  });

  return (
    <mesh ref={meshRef}>
      <cylinderGeometry args={[0.37, 0.37, 0.06, 64]} />
      <IrisMaterial />
    </mesh>
  );
}

// ============================================================================
// CORNEA MESH
// ============================================================================
function Cornea({ isExploded }: { isExploded: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const targetZ = useRef(0.95);

  useFrame((_state, delta) => {
    if (!meshRef.current) return;
    const target = isExploded ? 2.2 : 0.95;
    targetZ.current = THREE.MathUtils.lerp(targetZ.current, target, delta * 4);
    meshRef.current.position.z = targetZ.current;
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[1.02, 64, 64, 0, Math.PI * 2, 0, Math.PI * 0.35]} />
      <CorneaMaterial />
    </mesh>
  );
}

// ============================================================================
// LIGHTING SETUP - Cinematic 3-point lighting
// ============================================================================
function Lighting() {
  return (
    <>
      {/* Key light - warm, from upper right */}
      <directionalLight
        position={[5, 8, 5]}
        intensity={1.2}
        color="#FFF5E6"
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      {/* Fill light - cool, from lower left */}
      <directionalLight
        position={[-4, -2, 3]}
        intensity={0.4}
        color="#E6F0FF"
      />
      {/* Rim light - back, for edge definition */}
      <directionalLight
        position={[0, 3, -5]}
        intensity={0.6}
        color="#CCE5FF"
      />
      {/* Ambient - very subtle fill */}
      <ambientLight intensity={0.3} color="#F0F0FF" />
      {/* Inner glow from retina */}
      <pointLight position={[0, 0, -0.5]} color="#FF6B35" intensity={0.3} distance={3} />
    </>
  );
}

// ============================================================================
// TOOLTIP OVERLAY
// ============================================================================
const TOOLTIPS: Record<string, { title: string; desc: string }> = {
  Cornea: { title: 'Cornea', desc: 'Clear dome-shaped surface that refracts light. First focusing element.' },
  Iris: { title: 'Iris', desc: 'Colored muscular diaphragm controlling pupil size and light entry.' },
  Lens: { title: 'Crystalline Lens', desc: 'Flexible transparent structure providing fine focus adjustment.' },
  Retina: { title: 'Retina', desc: 'Light-sensitive neural tissue containing photoreceptors (rods & cones).' },
  'Optic Nerve': { title: 'Optic Nerve', desc: 'Cranial nerve II — transmits visual signals to the brain.' },
};

// ============================================================================
// MAIN EYE ASSEMBLY
// ============================================================================
function EyeAssembly() {
  const [isExploded, setIsExploded] = useState(false);
  const [hoveredPart, setHoveredPart] = useState<string | null>(null);
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (!groupRef.current) return;
    groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.15) * 0.25;
    groupRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.1) * 0.08;
  });

  return (
    <group ref={groupRef}>
      <Lighting />

      {/* Sclera (base) */}
      <Sclera isExploded={isExploded} />

      {/* Blood vessels on sclera */}
      <BloodVessels />

      {/* Cornea - outermost transparent layer */}
      <group
        onPointerOver={() => setHoveredPart('Cornea')}
        onPointerOut={() => setHoveredPart(null)}
        onClick={() => setIsExploded(!isExploded)}
      >
        <Cornea isExploded={isExploded} />
        {isExploded && hoveredPart === 'Cornea' && (
          <Html position={[0, 0, 2.5]} center>
            <div className="bg-gray-900/95 backdrop-blur-md border border-cyan-500/30 rounded-lg px-4 py-3 shadow-2xl max-w-[220px]">
              <h4 className="text-cyan-400 font-semibold text-sm mb-1">{TOOLTIPS.Cornea.title}</h4>
              <p className="text-gray-300 text-xs leading-relaxed">{TOOLTIPS.Cornea.desc}</p>
            </div>
          </Html>
        )}
      </group>

      {/* Iris with procedural texture */}
      <group
        onPointerOver={() => setHoveredPart('Iris')}
        onPointerOut={() => setHoveredPart(null)}
        onClick={() => setIsExploded(!isExploded)}
      >
        <Iris isExploded={isExploded} />
        {isExploded && hoveredPart === 'Iris' && (
          <Html position={[0.4, 0.3, 0.9]} center>
            <div className="bg-gray-900/95 backdrop-blur-md border border-cyan-500/30 rounded-lg px-4 py-3 shadow-2xl max-w-[220px]">
              <h4 className="text-cyan-400 font-semibold text-sm mb-1">{TOOLTIPS.Iris.title}</h4>
              <p className="text-gray-300 text-xs leading-relaxed">{TOOLTIPS.Iris.desc}</p>
            </div>
          </Html>
        )}
      </group>

      {/* Pupil */}
      <Pupil isExploded={isExploded} />

      {/* Lens */}
      <group
        onPointerOver={() => setHoveredPart('Lens')}
        onPointerOut={() => setHoveredPart(null)}
        onClick={() => setIsExploded(!isExploded)}
      >
        <Lens isExploded={isExploded} />
        {isExploded && hoveredPart === 'Lens' && (
          <Html position={[0.4, 0.2, 0.2]} center>
            <div className="bg-gray-900/95 backdrop-blur-md border border-cyan-500/30 rounded-lg px-4 py-3 shadow-2xl max-w-[220px]">
              <h4 className="text-cyan-400 font-semibold text-sm mb-1">{TOOLTIPS.Lens.title}</h4>
              <p className="text-gray-300 text-xs leading-relaxed">{TOOLTIPS.Lens.desc}</p>
            </div>
          </Html>
        )}
      </group>

      {/* Retina */}
      <group
        onPointerOver={() => setHoveredPart('Retina')}
        onPointerOut={() => setHoveredPart(null)}
        onClick={() => setIsExploded(!isExploded)}
      >
        <Retina isExploded={isExploded} isHovered={hoveredPart === 'Retina'} />
        {isExploded && hoveredPart === 'Retina' && (
          <Html position={[-0.4, 0.3, -1.0]} center>
            <div className="bg-gray-900/95 backdrop-blur-md border border-cyan-500/30 rounded-lg px-4 py-3 shadow-2xl max-w-[220px]">
              <h4 className="text-cyan-400 font-semibold text-sm mb-1">{TOOLTIPS.Retina.title}</h4>
              <p className="text-gray-300 text-xs leading-relaxed">{TOOLTIPS.Retina.desc}</p>
            </div>
          </Html>
        )}
      </group>

      {/* Optic Nerve */}
      <group
        onPointerOver={() => setHoveredPart('Optic Nerve')}
        onPointerOut={() => setHoveredPart(null)}
        onClick={() => setIsExploded(!isExploded)}
      >
        <OpticNerve isExploded={isExploded} isHovered={hoveredPart === 'Optic Nerve'} />
        {isExploded && hoveredPart === 'Optic Nerve' && (
          <Html position={[0.3, 0.2, -1.8]} center>
            <div className="bg-gray-900/95 backdrop-blur-md border border-cyan-500/30 rounded-lg px-4 py-3 shadow-2xl max-w-[220px]">
              <h4 className="text-cyan-400 font-semibold text-sm mb-1">{TOOLTIPS['Optic Nerve'].title}</h4>
              <p className="text-gray-300 text-xs leading-relaxed">{TOOLTIPS['Optic Nerve'].desc}</p>
            </div>
          </Html>
        )}
      </group>

      {/* Click instruction */}
      {!isExploded && (
        <Html position={[0, -1.3, 0]} center>
          <div className="text-cyan-400/60 text-xs uppercase tracking-widest animate-pulse">
            Click eye to explore anatomy
          </div>
        </Html>
      )}
    </group>
  );
}

// ============================================================================
// LOADING SPINNER
// ============================================================================
function LoadingSpinner() {
  return (
    <div className="w-full h-full min-h-[500px] flex items-center justify-center">
      <div className="text-center">
        <div className="w-16 h-16 border-4 border-cyan-500/20 border-t-cyan-500 rounded-full animate-spin mx-auto mb-4" />
        <p className="text-cyan-400/60 text-sm">Loading 3D Eye Model...</p>
      </div>
    </div>
  );
}

// ============================================================================
// MAIN EXPORT — Canvas wrapped in Suspense
// ============================================================================
export default function EyeModel3D() {
  return (
    <div className="w-full h-full min-h-[500px]">
      <Suspense fallback={<LoadingSpinner />}>
        <Canvas
          camera={{ position: [0, 0, 3.2], fov: 40 }}
          gl={{
            antialias: true,
            alpha: true,
            powerPreference: 'high-performance',
            failIfMajorPerformanceCaveat: false,
          }}
          dpr={[1, 1.5]}
          onCreated={({ gl }) => {
            gl.toneMapping = THREE.ACESFilmicToneMapping;
            gl.toneMappingExposure = 1.2;
          }}
        >
          <color attach="background" args={['transparent']} />
          <fog attach="fog" args={['#0B0F19', 6, 12]} />

          <EyeAssembly />

          <OrbitControls
            enablePan={false}
            enableZoom={true}
            minDistance={2}
            maxDistance={6}
            autoRotate={false}
            enableDamping
            dampingFactor={0.05}
            maxPolarAngle={Math.PI * 0.75}
            minPolarAngle={Math.PI * 0.25}
          />

          <ContactShadows
            position={[0, -1.4, 0]}
            opacity={0.25}
            scale={5}
            blur={2.5}
            far={3}
          />

          <Environment preset="studio" />
        </Canvas>
      </Suspense>
    </div>
  );
}
