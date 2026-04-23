import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Eye, Users, Target, Award, Globe } from 'lucide-react';
import { Button } from '@/components/ui/button';

const team = [
  { name:'Dr. Aryan Mehta',    role:'Co-Founder & CEO',      bg:'from-cyan-500 to-blue-600',    init:'AM' },
  { name:'Dr. Priya Sharma',   role:'Chief Medical Officer',  bg:'from-purple-500 to-pink-600',  init:'PS' },
  { name:'Rahul Verma',        role:'CTO & ML Lead',          bg:'from-emerald-500 to-teal-600', init:'RV' },
  { name:'Sneha Kapoor',       role:'Head of Design',         bg:'from-orange-500 to-red-500',   init:'SK' },
];

const milestones = [
  { year:'2022', text:'Founded in IIT Bombay with a mission to democratize eye care with AI.' },
  { year:'2023', text:'First clinical study — 94% accuracy on 10,000 retinal images.' },
  { year:'2024', text:'Partnership with 15 major eye hospitals across India.' },
  { year:'2025', text:'Launched multi-disease platform covering 6 conditions.' },
  { year:'2026', text:'377,000 image training set — targeting global deployment.' },
];

export default function AboutPage() {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <Button variant="ghost" size="sm" onClick={() => navigate('/')} className="mb-8 text-muted-foreground">
          <ArrowLeft className="w-4 h-4 mr-2" />Back to Home
        </Button>

        <motion.div initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-primary/10 rounded-xl flex items-center justify-center">
              <Eye className="w-5 h-5 text-primary" />
            </div>
            <span className="text-sm text-primary font-medium uppercase tracking-wider">About Us</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-6">
            Our Mission: <span className="text-primary">See the Unseen</span>
          </h1>
          <p className="text-lg text-muted-foreground max-w-3xl mb-16 leading-relaxed">
            EYE-ASSISST is an AI-powered retinal disease screening platform built to make world-class ophthalmological diagnostics accessible to everyone — from major hospitals to remote clinics.
          </p>
        </motion.div>

        {/* Values */}
        <div className="grid md:grid-cols-3 gap-6 mb-20">
          {[
            { icon:Target, title:'Precision First', text:'Our models achieve AUC scores of 0.91–0.97 across 6 disease classes, validated on 377,000+ images.' },
            { icon:Globe,  title:'Global Access',  text:'Designed for use anywhere — from urban hospitals to rural telemedicine setups.' },
            { icon:Award,  title:'Clinical Trust',  text:'Co-designed with ophthalmologists and validated in real clinical settings.' },
          ].map((v,i) => (
            <motion.div key={v.title} initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} transition={{ delay:i*0.1 }}
              className="p-6 rounded-2xl border border-border bg-card">
              <v.icon className="w-8 h-8 text-primary mb-4" />
              <h3 className="font-semibold text-foreground mb-2">{v.title}</h3>
              <p className="text-sm text-muted-foreground">{v.text}</p>
            </motion.div>
          ))}
        </div>

        {/* Team */}
        <h2 className="text-2xl font-bold text-foreground mb-8 flex items-center gap-2">
          <Users className="w-6 h-6 text-primary" />Meet the Team
        </h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5 mb-20">
          {team.map((t,i) => (
            <motion.div key={t.name} initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} transition={{ delay:i*0.1 }}
              className="p-5 rounded-2xl border border-border bg-card text-center">
              <div className={`w-16 h-16 rounded-full bg-gradient-to-br ${t.bg} flex items-center justify-center text-2xl font-bold text-white mx-auto mb-3`}>{t.init}</div>
              <p className="font-semibold text-foreground">{t.name}</p>
              <p className="text-xs text-muted-foreground mt-1">{t.role}</p>
            </motion.div>
          ))}
        </div>

        {/* Timeline */}
        <h2 className="text-2xl font-bold text-foreground mb-8">Our Journey</h2>
        <div className="space-y-4">
          {milestones.map((m,i) => (
            <motion.div key={m.year} initial={{ opacity:0, x:-20 }} animate={{ opacity:1, x:0 }} transition={{ delay:i*0.08 }}
              className="flex gap-5 items-start">
              <span className="text-primary font-bold font-mono w-12 flex-shrink-0">{m.year}</span>
              <div className="bg-card border border-border rounded-xl p-4 flex-1">
                <p className="text-sm text-muted-foreground">{m.text}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
