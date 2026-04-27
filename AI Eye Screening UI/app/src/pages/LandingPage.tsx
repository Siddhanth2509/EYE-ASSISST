import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import type { LucideIcon } from 'lucide-react';
import {
  Eye,
  Upload,
  Brain,
  FileText,
  ChevronRight,
  Shield,
  Zap,
  Users,
  BarChart3,
  Activity,
  ArrowRight,
  Microscope,
  Clock,
  Award,
  Globe,
  Sparkles,
  Check,
  CheckCircle,
} from 'lucide-react';
import PaymentGateway from '@/components/PaymentGateway';
import { Button } from '@/components/ui/button';
import { Suspense, lazy } from 'react';
const EyeModel3D = lazy(() => import('@/components/three/EyeModel3D'));

gsap.registerPlugin(ScrollTrigger);

// Animated counter hook
function useCountUp(end: number, duration: number = 2, start: boolean = false) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!start) return;
    let startTime: number;
    let animationFrame: number;

    const animate = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / (duration * 1000), 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease out cubic
      setCount(Math.floor(eased * end));
      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate);
      }
    };

    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, [end, duration, start]);

  return count;
}

// Particle Field Background
function ParticleField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    const particles: Array<{
      x: number;
      y: number;
      vx: number;
      vy: number;
      size: number;
      opacity: number;
    }> = [];

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    // Create particles
    for (let i = 0; i < 60; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        size: Math.random() * 2 + 1,
        opacity: Math.random() * 0.5 + 0.2,
      });
    }

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(34, 211, 238, ${p.opacity})`;
        ctx.fill();
      });

      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 150) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(34, 211, 238, ${0.1 * (1 - dist / 150)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
      style={{ zIndex: 1 }}
    />
  );
}

// Stat Card with counter
function StatCard({
  icon: Icon,
  value,
  suffix,
  label,
  delay,
}: {
  icon: LucideIcon;
  value: number;
  suffix: string;
  label: string;
  delay: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);
  const count = useCountUp(value, 2, isVisible);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setTimeout(() => setIsVisible(true), delay * 1000);
          observer.disconnect();
        }
      },
      { threshold: 0.5 }
    );

    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [delay]);

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay, duration: 0.6 }}
      className="text-center"
    >
      <div className="w-14 h-14 bg-primary/10 rounded-xl flex items-center justify-center mx-auto mb-4">
        <Icon className="w-7 h-7 text-primary" />
      </div>
      <div className="text-4xl md:text-5xl font-bold mb-2">
        <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
          {count.toLocaleString()}
        </span>
        <span className="text-primary">{suffix}</span>
      </div>
      <p className="text-muted-foreground text-sm">{label}</p>
    </motion.div>
  );
}

// Disease Card with flip effect
function DiseaseCard({
  name,
  fullName,
  auc,
  description,
  color,
  delay,
}: {
  name: string;
  fullName: string;
  auc: string;
  description: string;
  color: string;
  delay: number;
}) {
  const [isFlipped, setIsFlipped] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, rotateY: 90 }}
      whileInView={{ opacity: 1, rotateY: 0 }}
      viewport={{ once: true }}
      transition={{ delay, duration: 0.6, type: 'spring' }}
      className="perspective-[1000px]"
      onMouseEnter={() => setIsFlipped(true)}
      onMouseLeave={() => setIsFlipped(false)}
    >
      <div
        className="relative w-full h-64 transition-transform duration-500 preserve-3d"
        style={{
          transformStyle: 'preserve-3d',
          transform: isFlipped ? 'rotateY(180deg)' : 'rotateY(0deg)',
        }}
      >
        {/* Front */}
        <div
          className="absolute inset-0 backface-hidden rounded-xl border border-border bg-card p-6 flex flex-col items-center justify-center"
          style={{ backfaceVisibility: 'hidden' }}
        >
          <div
            className="w-16 h-16 rounded-full flex items-center justify-center mb-4"
            style={{ backgroundColor: `${color}20`, border: `2px solid ${color}40` }}
          >
            <Eye className="w-8 h-8" style={{ color }} />
          </div>
          <h3 className="text-xl font-bold mb-1">{name}</h3>
          <p className="text-sm text-muted-foreground">{fullName}</p>
          <div className="mt-4 flex items-center gap-1 text-xs text-muted-foreground">
            <Sparkles className="w-3 h-3" />
            Hover to learn more
          </div>
        </div>

        {/* Back */}
        <div
          className="absolute inset-0 rounded-xl border p-6 flex flex-col items-center justify-center text-center"
          style={{
            backfaceVisibility: 'hidden',
            transform: 'rotateY(180deg)',
            backgroundColor: `${color}10`,
            borderColor: `${color}30`,
          }}
        >
          <div className="text-3xl font-bold mb-2" style={{ color }}>
            {auc}
          </div>
          <p className="text-sm text-muted-foreground mb-3">AUC Score</p>
          <p className="text-xs leading-relaxed">{description}</p>
        </div>
      </div>
    </motion.div>
  );
}

// How it works step
function StepCard({
  number,
  icon: Icon,
  title,
  description,
  delay,
}: {
  number: string;
  icon: LucideIcon;
  title: string;
  description: string;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -30 }}
      whileInView={{ opacity: 1, x: 0 }}
      viewport={{ once: true }}
      transition={{ delay, duration: 0.6 }}
      className="flex items-start gap-4"
    >
      <div className="flex-shrink-0 w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center">
        <span className="text-lg font-bold text-primary">{number}</span>
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2 mb-2">
          <Icon className="w-5 h-5 text-primary" />
          <h3 className="text-lg font-semibold">{title}</h3>
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed">{description}</p>
      </div>
    </motion.div>
  );
}

// Feature card
function FeatureCard({
  icon: Icon,
  title,
  description,
  delay,
}: {
  icon: LucideIcon;
  title: string;
  description: string;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay, duration: 0.5 }}
      className="group p-6 rounded-xl border border-border bg-card/50 hover:bg-card hover:border-primary/30 transition-all duration-300"
    >
      <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
        <Icon className="w-5 h-5 text-primary" />
      </div>
      <h3 className="font-semibold mb-2">{title}</h3>
      <p className="text-sm text-muted-foreground leading-relaxed">{description}</p>
    </motion.div>
  );
}

// Main Landing Page Component
export default function LandingPage() {
  const navigate = useNavigate();
  const heroRef = useRef<HTMLDivElement>(null);
  const [scrolled, setScrolled] = useState(false);
  const [billing, setBilling] = useState<'monthly'|'annual'>('monthly');
  const [showPayment, setShowPayment] = useState(false);
  const [payCtx, setPayCtx] = useState<{amount:number;desc:string}|null>(null);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // GSAP ScrollTrigger animations
  useEffect(() => {
    const ctx = gsap.context(() => {
      // Hero text reveal
      gsap.from('.hero-title span', {
        y: 80,
        opacity: 0,
        duration: 1,
        stagger: 0.1,
        ease: 'power3.out',
      });

      gsap.from('.hero-subtitle', {
        y: 40,
        opacity: 0,
        duration: 0.8,
        delay: 0.6,
        ease: 'power3.out',
      });

      gsap.from('.hero-cta', {
        y: 30,
        opacity: 0,
        duration: 0.8,
        delay: 0.9,
        ease: 'power3.out',
      });
    }, heroRef);

    return () => ctx.revert();
  }, []);

  const stats = [
    { icon: Activity, value: 50000, suffix: '+', label: 'Scans Analyzed' },
    { icon: Microscope, value: 6, suffix: '', label: 'Disease Types' },
    { icon: Award, value: 97, suffix: '%', label: 'AUC Score' },
    { icon: Users, value: 1200, suffix: '+', label: 'Medical Professionals' },
  ];

  const diseases = [
    {
      name: 'DR',
      fullName: 'Diabetic Retinopathy',
      auc: '0.97',
      description: 'Detects microaneurysms, hemorrhages, and neovascularization from diabetes.',
      color: '#EF4444',
    },
    {
      name: 'AMD',
      fullName: 'Age-related Macular Degeneration',
      auc: '0.95',
      description: 'Identifies drusen, geographic atrophy, and choroidal neovascularization.',
      color: '#F59E0B',
    },
    {
      name: 'Glaucoma',
      fullName: 'Open-angle Glaucoma',
      auc: '0.94',
      description: 'Assesses optic nerve cupping, rim thinning, and RNFL defects.',
      color: '#8B5CF6',
    },
    {
      name: 'Cataract',
      fullName: 'Cortical Cataract',
      auc: '0.93',
      description: 'Detects lens opacities, cortical spokes, and posterior subcapsular changes.',
      color: '#3B82F6',
    },
    {
      name: 'RVO',
      fullName: 'Retinal Vein Occlusion',
      auc: '0.92',
      description: 'Identifies vessel tortuosity, dot-blot hemorrhages, and macular edema.',
      color: '#10B981',
    },
    {
      name: 'Myopia',
      fullName: 'Pathological Myopia',
      auc: '0.91',
      description: 'Detects tessellation, lacquer cracks, and posterior staphyloma.',
      color: '#EC4899',
    },
  ];

  const features = [
    {
      icon: Brain,
      title: 'Deep Learning AI',
      description: 'State-of-the-art convolutional neural networks trained on millions of retinal images.',
    },
    {
      icon: Zap,
      title: 'Real-time Analysis',
      description: 'Get AI-powered results in under 10 seconds with confidence scores for each finding.',
    },
    {
      icon: Shield,
      title: 'HIPAA Compliant',
      description: 'Enterprise-grade security with end-to-end encryption and full audit trails.',
    },
    {
      icon: FileText,
      title: 'Automated Reports',
      description: 'Generate clinical-grade PDF reports with Grad-CAM visualizations automatically.',
    },
    {
      icon: Globe,
      title: 'Multi-site Access',
      description: 'Cloud-based platform accessible from any device, anywhere in the world.',
    },
    {
      icon: Clock,
      title: '24/7 Screening',
      description: 'Round-the-clock automated screening with no waiting times or scheduling.',
    },
  ];

  return (
    <div className="min-h-screen bg-background overflow-x-hidden">
      {/* Navigation */}
      <nav
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
          scrolled ? 'bg-background/90 backdrop-blur-lg border-b border-border' : 'bg-transparent'
        }`}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
                <Eye className="w-5 h-5 text-primary" />
              </div>
              <span className="font-bold text-lg tracking-tight">EYE-ASSISST</span>
            </div>
            <div className="hidden md:flex items-center gap-6">
              <a href="#features" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                Features
              </a>
              <a href="#diseases" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                Diseases
              </a>
              <a href="#how-it-works" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                How it Works
              </a>
              <Button size="sm" onClick={() => navigate('/login')} className="btn-medical">
                Start Screening
                <ArrowRight className="w-4 h-4 ml-1" />
              </Button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section ref={heroRef} className="relative min-h-screen flex items-center pt-16">
        <ParticleField />

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Text Content */}
            <div className="order-2 lg:order-1">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 rounded-full mb-6"
              >
                <Sparkles className="w-4 h-4 text-primary" />
                <span className="text-sm text-primary font-medium">
                  AI-Powered Retinal Screening
                </span>
              </motion.div>

              <h1 className="hero-title text-4xl md:text-5xl lg:text-6xl font-bold leading-tight mb-6">
                <span className="inline-block">See Beyond</span>{' '}
                <span className="inline-block">the Surface</span>
                <br />
                <span className="inline-block bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 bg-clip-text text-transparent">
                  AI Eye Disease Screening
                </span>
              </h1>

              <p className="hero-subtitle text-lg text-muted-foreground mb-8 max-w-lg leading-relaxed">
                Advanced deep learning technology for early detection of diabetic retinopathy,
                glaucoma, AMD, and more. Trusted by 1,200+ ophthalmologists worldwide.
              </p>

              <div className="hero-cta flex flex-wrap gap-4">
                <Button
                  size="lg"
                  onClick={() => navigate('/login')}
                  className="btn-medical text-lg px-8"
                >
                  Start Screening
                  <ChevronRight className="w-5 h-5 ml-1" />
                </Button>
                <Button
                  size="lg"
                  variant="outline"
                  onClick={() => document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' })}
                  className="text-lg px-8"
                >
                  Learn More
                </Button>
              </div>

              {/* Trust badges */}
              <div className="mt-12 flex items-center gap-6">
                <div className="flex -space-x-2">
                  {[1, 2, 3, 4].map((i) => (
                    <div
                      key={i}
                      className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-400 to-blue-500 border-2 border-background flex items-center justify-center text-xs font-bold"
                    >
                      {String.fromCharCode(64 + i)}
                    </div>
                  ))}
                </div>
                <div>
                  <p className="text-sm font-medium">Trusted by leading hospitals</p>
                  <p className="text-xs text-muted-foreground">Mayo Clinic, Johns Hopkins, Moorfields</p>
                </div>
              </div>
            </div>

            {/* 3D Eye */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 1, delay: 0.3 }}
              className="order-1 lg:order-2 h-[400px] lg:h-[600px] rounded-3xl overflow-hidden"
              style={{ background: 'radial-gradient(ellipse at center, #0d1a2e 0%, #0B0F19 100%)' }}
            >
              <Suspense fallback={<div className="w-full h-full flex items-center justify-center text-muted-foreground"><Activity className="w-8 h-8 animate-pulse" /></div>}>
                <EyeModel3D />
              </Suspense>
            </motion.div>
          </div>
        </div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <div className="w-6 h-10 border-2 border-primary/30 rounded-full flex justify-center pt-2">
            <motion.div
              animate={{ y: [0, 12, 0] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="w-1.5 h-1.5 bg-primary rounded-full"
            />
          </div>
        </motion.div>
      </section>

      {/* Stats Section */}
      <section className="py-20 border-y border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-12">
            {stats.map((stat, i) => (
              <StatCard key={stat.label} {...stat} delay={i * 0.15} />
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Why <span className="text-primary">EYE-ASSISST</span>?
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Comprehensive AI screening platform designed for modern ophthalmology practices
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, i) => (
              <FeatureCard key={feature.title} {...feature} delay={i * 0.1} />
            ))}
          </div>
        </div>
      </section>

      {/* Disease Detection Section */}
      <section id="diseases" className="py-24 bg-muted/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              6 Diseases, <span className="text-primary">One Platform</span>
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              State-of-the-art detection for the most common sight-threatening conditions
            </p>
          </motion.div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {diseases.map((disease, i) => (
              <DiseaseCard key={disease.name} {...disease} delay={i * 0.1} />
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              How It <span className="text-primary">Works</span>
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              From image upload to clinical report in under 60 seconds
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            <StepCard
              number="1"
              icon={Upload}
              title="Upload Fundus Image"
              description="Drag and drop a retinal fundus photograph. Supports JPEG, PNG, and DICOM formats. Automatic quality validation ensures optimal analysis."
              delay={0}
            />
            <StepCard
              number="2"
              icon={Brain}
              title="AI Analysis"
              description="Our deep learning model analyzes the image in real-time, detecting 6 diseases with explainable Grad-CAM heatmaps showing exactly where the AI is looking."
              delay={0.2}
            />
            <StepCard
              number="3"
              icon={FileText}
              title="Clinical Report"
              description="Receive a comprehensive PDF report with AI findings, confidence scores, and recommendations. Doctor review and approval workflow included."
              delay={0.4}
            />
          </div>

          {/* Animated flow connector */}
          <div className="hidden md:flex justify-center mt-8">
            <motion.div
              initial={{ width: 0 }}
              whileInView={{ width: '60%' }}
              viewport={{ once: true }}
              transition={{ duration: 1.5, delay: 0.5 }}
              className="h-0.5 bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500"
            />
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div initial={{ opacity:0, y:20 }} whileInView={{ opacity:1, y:0 }} viewport={{ once:true }} className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Simple, <span className="text-primary">Transparent</span> Pricing</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto mb-8">Choose the plan that fits your practice. No hidden fees.</p>
            {/* Billing toggle */}
            <div className="inline-flex items-center gap-3 bg-card border border-border rounded-full p-1">
              <button onClick={() => setBilling('monthly')} className={`px-5 py-2 rounded-full text-sm font-medium transition-all ${billing==='monthly' ? 'bg-primary text-primary-foreground':'text-muted-foreground hover:text-foreground'}`}>Monthly</button>
              <button onClick={() => setBilling('annual')}  className={`px-5 py-2 rounded-full text-sm font-medium transition-all ${billing==='annual'  ? 'bg-primary text-primary-foreground':'text-muted-foreground hover:text-foreground'}`}>
                Annual <span className="text-xs text-emerald-400 ml-1">−20%</span>
              </button>
            </div>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6">
            {/* Free */}
            <motion.div initial={{ opacity:0, y:30 }} whileInView={{ opacity:1, y:0 }} viewport={{ once:true }} transition={{ delay:0 }}
              className="rounded-2xl border border-border bg-card p-8 flex flex-col">
              <p className="text-sm font-medium text-muted-foreground mb-2">Free</p>
              <div className="mb-6">
                <span className="text-5xl font-bold text-foreground">₹0</span>
                <span className="text-muted-foreground ml-1">/month</span>
              </div>
              <ul className="space-y-3 mb-8 flex-1">
                {['5 scans per day','Basic DR screening','PDF report download','Email support'].map(f=><li key={f} className="flex items-center gap-2 text-sm text-muted-foreground"><Check className="w-4 h-4 text-emerald-400 flex-shrink-0"/>{f}</li>)}
              </ul>
              <Button variant="outline" className="w-full" onClick={() => navigate('/login')}>Get Started Free</Button>
            </motion.div>

            {/* Pro — highlighted */}
            <motion.div initial={{ opacity:0, y:30 }} whileInView={{ opacity:1, y:0 }} viewport={{ once:true }} transition={{ delay:0.1 }}
              className="rounded-2xl border-2 border-primary bg-primary/5 p-8 flex flex-col relative shadow-[0_0_40px_rgba(34,211,238,0.15)]">
              <div className="absolute -top-4 left-1/2 -translate-x-1/2"><span className="bg-primary text-primary-foreground text-xs font-bold px-4 py-1.5 rounded-full">Most Popular</span></div>
              <p className="text-sm font-medium text-primary mb-2">Pro</p>
              <div className="mb-6">
                {billing==='annual' && <div className="text-sm text-muted-foreground line-through">₹999/mo</div>}
                <span className="text-5xl font-bold text-foreground">{billing==='annual' ? '₹799':'₹999'}</span>
                <span className="text-muted-foreground ml-1">/month</span>
              </div>
              <ul className="space-y-3 mb-8 flex-1">
                {['Unlimited scans','All 6 disease detection','Grad-CAM heatmaps','Patient portal access','Appointment booking','Priority support'].map(f=><li key={f} className="flex items-center gap-2 text-sm text-foreground"><CheckCircle className="w-4 h-4 text-primary flex-shrink-0"/>{f}</li>)}
              </ul>
              <Button className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700"
                onClick={() => { setPayCtx({ amount: billing==='annual'?799:999, desc:'EYE-ASSISST Pro Subscription' }); setShowPayment(true); }}>
                Start Pro Plan
              </Button>
            </motion.div>

            {/* Enterprise */}
            <motion.div initial={{ opacity:0, y:30 }} whileInView={{ opacity:1, y:0 }} viewport={{ once:true }} transition={{ delay:0.2 }}
              className="rounded-2xl border border-border bg-card p-8 flex flex-col">
              <p className="text-sm font-medium text-muted-foreground mb-2">Enterprise</p>
              <div className="mb-6">
                <span className="text-4xl font-bold text-foreground">Custom</span>
              </div>
              <ul className="space-y-3 mb-8 flex-1">
                {['Everything in Pro','REST API access','White-label branding','SLA guarantee','Dedicated support','HIPAA compliance docs'].map(f=><li key={f} className="flex items-center gap-2 text-sm text-muted-foreground"><Check className="w-4 h-4 text-emerald-400 flex-shrink-0"/>{f}</li>)}
              </ul>
              <Button variant="outline" className="w-full" onClick={() => navigate('/contact')}>Contact Sales</Button>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-gradient-to-br from-primary/5 via-background to-primary/5 border-y border-border">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-3xl md:text-5xl font-bold mb-6">
              Ready to Transform Your
              <span className="block bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent mt-2">
                Screening Workflow?
              </span>
            </h2>
            <p className="text-lg text-muted-foreground mb-8 max-w-2xl mx-auto">
              Join thousands of ophthalmologists who trust EYE-ASSISST for faster,
              more accurate retinal disease detection.
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <Button
                size="lg"
                onClick={() => navigate('/login')}
                className="btn-medical text-lg px-10 py-6"
              >
                Get Started Free
                <ArrowRight className="w-5 h-5 ml-2" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="text-lg px-10 py-6"
                onClick={() => navigate('/demo')}
              >
                <BarChart3 className="w-5 h-5 mr-2" />
                View Demo
              </Button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-4 gap-8 mb-8">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
                  <Eye className="w-5 h-5 text-primary" />
                </div>
                <span className="font-bold">EYE-ASSISST</span>
              </div>
              <p className="text-sm text-muted-foreground">
                AI-powered retinal disease screening for modern ophthalmology.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><a href="#features" className="hover:text-primary transition-colors">Features</a></li>
                <li><a href="#diseases" className="hover:text-primary transition-colors">Diseases</a></li>
                <li><button onClick={()=>document.getElementById('pricing')?.scrollIntoView({behavior:'smooth'})} className="hover:text-primary transition-colors">Pricing</button></li>
                <li><button onClick={()=>navigate('/docs')} className="hover:text-primary transition-colors">API Docs</button></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Resources</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><button onClick={()=>navigate('/docs')}          className="hover:text-primary transition-colors">Documentation</button></li>
                <li><button onClick={()=>navigate('/research')}      className="hover:text-primary transition-colors">Research</button></li>
                <li><button onClick={()=>navigate('/case-studies')}  className="hover:text-primary transition-colors">Case Studies</button></li>
                <li><button onClick={()=>navigate('/blog')}          className="hover:text-primary transition-colors">Blog</button></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><button onClick={()=>navigate('/about')}    className="hover:text-primary transition-colors">About</button></li>
                <li><button onClick={()=>navigate('/careers')}  className="hover:text-primary transition-colors">Careers</button></li>
                <li><button onClick={()=>navigate('/contact')}  className="hover:text-primary transition-colors">Contact</button></li>
                <li><button onClick={()=>navigate('/privacy')}  className="hover:text-primary transition-colors">Privacy</button></li>
              </ul>
            </div>
          </div>
          <div className="pt-8 border-t border-border flex flex-col md:flex-row items-center justify-between gap-4">
            <p className="text-sm text-muted-foreground">
              &copy; 2026 EYE-ASSISST. All rights reserved.
            </p>
            <div className="flex items-center gap-4">
              <Shield className="w-4 h-4 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">HIPAA Compliant • SOC 2 Certified</span>
            </div>
          </div>
        </div>
      </footer>

      {/* Payment Gateway (from pricing section) */}
      {showPayment && payCtx && (
        <PaymentGateway
          amount={payCtx.amount}
          description={payCtx.desc}
          onSuccess={() => { setShowPayment(false); setPayCtx(null); }}
          onClose={() => { setShowPayment(false); setPayCtx(null); }}
        />
      )}
    </div>
  );
}
