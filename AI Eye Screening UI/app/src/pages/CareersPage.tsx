import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Briefcase, MapPin, Clock, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

const jobs = [
  { id:1, title:'Senior ML Engineer — Retinal AI',  team:'Research',    location:'Mumbai / Remote', type:'Full-time', desc:'Build and train next-gen ResNet/ViT models on 500k+ retinal images. Experience with PyTorch, Albumentations, and medical imaging required.' },
  { id:2, title:'Full-Stack Engineer (React + FastAPI)', team:'Engineering', location:'Remote',         type:'Full-time', desc:'Own the frontend and backend of EYE-ASSISST. React 19, TypeScript, Python, FastAPI, WebSockets.' },
  { id:3, title:'Clinical Research Coordinator',  team:'Medical',     location:'Mumbai',          type:'Full-time', desc:'Work with ophthalmologists to validate AI outputs and manage IRB-approved clinical studies.' },
];

export default function CareersPage() {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <Button variant="ghost" size="sm" onClick={() => navigate('/')} className="mb-8 text-muted-foreground">
          <ArrowLeft className="w-4 h-4 mr-2" />Back to Home
        </Button>

        <motion.div initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} className="mb-12">
          <div className="flex items-center gap-2 mb-3">
            <Briefcase className="w-5 h-5 text-primary" />
            <span className="text-sm text-primary font-medium">Careers</span>
          </div>
          <h1 className="text-4xl font-bold mb-3">Join Our <span className="text-primary">Mission</span></h1>
          <p className="text-muted-foreground text-lg max-w-2xl">
            We're a small team with outsized impact. If you care about using AI to prevent blindness, we want to hear from you.
          </p>
        </motion.div>

        {/* Perks */}
        <div className="grid sm:grid-cols-3 gap-4 mb-12">
          {[
            { label:'Remote First',    text:'Work from anywhere in India or globally.' },
            { label:'Meaningful Work', text:'Your code helps doctors catch blindness early.' },
            { label:'Equity + ESOP',   text:'Early team members get significant equity.' },
          ].map((p,i) => (
            <motion.div key={p.label} initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} transition={{ delay:i*0.07 }}
              className="p-5 rounded-2xl border border-border bg-card">
              <p className="font-semibold text-foreground mb-1">{p.label}</p>
              <p className="text-sm text-muted-foreground">{p.text}</p>
            </motion.div>
          ))}
        </div>

        {/* Open Roles */}
        <h2 className="text-xl font-semibold text-foreground mb-5">Open Positions ({jobs.length})</h2>
        <div className="space-y-4">
          {jobs.map((j,i) => (
            <motion.div key={j.id} initial={{ opacity:0, x:-20 }} animate={{ opacity:1, x:0 }} transition={{ delay:i*0.08 }}
              className="p-6 rounded-2xl border border-border bg-card hover:border-primary/30 transition-colors group cursor-pointer">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors">{j.title}</h3>
                  <div className="flex items-center gap-4 mt-2">
                    <span className="text-xs bg-primary/10 text-primary px-2 py-1 rounded-full">{j.team}</span>
                    <span className="flex items-center gap-1 text-xs text-muted-foreground"><MapPin className="w-3 h-3"/>{j.location}</span>
                    <span className="flex items-center gap-1 text-xs text-muted-foreground"><Clock className="w-3 h-3"/>{j.type}</span>
                  </div>
                  <p className="text-sm text-muted-foreground mt-3">{j.desc}</p>
                </div>
                <ChevronRight className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors flex-shrink-0 mt-1" />
              </div>
            </motion.div>
          ))}
        </div>

        <div className="mt-10 p-6 rounded-2xl bg-primary/5 border border-primary/20 text-center">
          <p className="text-foreground font-medium mb-1">Don't see a fit?</p>
          <p className="text-sm text-muted-foreground mb-4">Send us your best work at <span className="text-primary">careers@eyeassist.ai</span></p>
          <Button onClick={() => navigate('/contact')}>Contact Us</Button>
        </div>
      </div>
    </div>
  );
}
