import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Briefcase, MapPin, Clock, ChevronRight, ChevronDown, X, Mail } from 'lucide-react';
import { Button } from '@/components/ui/button';

const jobs = [
  {
    id: 1,
    title: 'Senior ML Engineer — Retinal AI',
    team: 'Research',
    location: 'Mumbai / Remote',
    type: 'Full-time',
    desc: 'Build and train next-gen ResNet/ViT models on 500k+ retinal images. Experience with PyTorch, Albumentations, and medical imaging required.',
    fullDesc: `We're looking for a senior ML engineer to own the model development lifecycle for our retinal disease detection platform.

**Responsibilities:**
• Design and train multi-label classification models across 6+ disease classes
• Implement explainability techniques (Grad-CAM, LIME) for clinical trust
• Optimize inference pipeline for sub-200ms predictions on GPU and CPU
• Collaborate with ophthalmologists for clinical validation and ground truth labeling
• Publish research findings and contribute to regulatory submissions

**Requirements:**
• 4+ years of ML engineering experience, preferably in medical imaging
• Deep expertise in PyTorch, torchvision, and modern CNN/ViT architectures
• Experience with class-imbalanced datasets and techniques (Focal Loss, oversampling)
• Familiarity with DICOM, fundus photography, and ophthalmic datasets
• Strong background in model evaluation (AUC, sensitivity, specificity)

**Nice to have:** Publication record, experience with FDA/CDSCO SaMD submissions.`,
  },
  {
    id: 2,
    title: 'Full-Stack Engineer (React + FastAPI)',
    team: 'Engineering',
    location: 'Remote',
    type: 'Full-time',
    desc: 'Own the frontend and backend of EYE-ASSISST. React 19, TypeScript, Python, FastAPI, WebSockets.',
    fullDesc: `You'll be the engineering foundation of EYE-ASSISST, building the systems that clinicians and researchers use daily.

**Responsibilities:**
• Develop and maintain the React + TypeScript frontend (dashboard, patient portal, analytics)
• Build and optimize FastAPI backend endpoints for image analysis and report generation
• Design real-time features using WebSockets for live scan tracking
• Implement secure authentication (OAuth2, JWT) and role-based access control
• Set up CI/CD pipelines and cloud deployment on AWS/GCP

**Requirements:**
• 3+ years of full-stack experience
• Proficiency in React 18+, TypeScript, and modern CSS frameworks
• Strong Python skills with FastAPI or similar async frameworks
• Experience with PostgreSQL, Redis, and cloud storage (S3/GCS)
• Understanding of HIPAA/DPDPA compliance in healthcare applications

**Nice to have:** Experience with Three.js, WebGL, or medical imaging frontends.`,
  },
  {
    id: 3,
    title: 'Clinical Research Coordinator',
    team: 'Medical',
    location: 'Mumbai',
    type: 'Full-time',
    desc: 'Work with ophthalmologists to validate AI outputs and manage IRB-approved clinical studies.',
    fullDesc: `Bridge the gap between our AI team and clinical partners at leading ophthalmology centers across India.

**Responsibilities:**
• Coordinate multi-site clinical validation studies for AI models
• Manage IRB/IEC submissions and ethics approvals
• Collect and annotate fundus images with ophthalmologist ground truth labels
• Write clinical study protocols, CRFs, and regulatory documentation
• Liaise with AIIMS, LV Prasad, and Sankara Nethralaya partner hospitals

**Requirements:**
• MBBS or MSc in a life sciences field
• 2+ years experience in clinical research or healthcare IT
• Familiarity with GCP (Good Clinical Practice) guidelines
• Strong written and verbal communication skills
• Ability to travel to partner hospital sites in Mumbai, Chennai, Hyderabad

**Nice to have:** Experience with retinal imaging, NPCB programs, or CDSCO regulatory submissions.`,
  },
];

const teamColors: Record<string, string> = {
  Research: 'bg-cyan-500/10 text-cyan-400',
  Engineering: 'bg-purple-500/10 text-purple-400',
  Medical: 'bg-emerald-500/10 text-emerald-400',
};

export default function CareersPage() {
  const navigate = useNavigate();
  const [expandedJob, setExpandedJob] = useState<number | null>(null);
  const [applyJob, setApplyJob] = useState<typeof jobs[0] | null>(null);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <Button variant="ghost" size="sm" onClick={() => navigate('/')} className="mb-8 text-muted-foreground">
          <ArrowLeft className="w-4 h-4 mr-2" />Back to Home
        </Button>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-12">
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
            { label: 'Remote First', text: 'Work from anywhere in India or globally.' },
            { label: 'Meaningful Work', text: 'Your code helps doctors catch blindness early.' },
            { label: 'Equity + ESOP', text: 'Early team members get significant equity.' },
          ].map((p, i) => (
            <motion.div key={p.label} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.07 }}
              className="p-5 rounded-2xl border border-border bg-card">
              <p className="font-semibold text-foreground mb-1">{p.label}</p>
              <p className="text-sm text-muted-foreground">{p.text}</p>
            </motion.div>
          ))}
        </div>

        {/* Open Roles */}
        <h2 className="text-xl font-semibold text-foreground mb-5">Open Positions ({jobs.length})</h2>
        <div className="space-y-4">
          {jobs.map((j, i) => (
            <motion.div key={j.id} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.08 }}>
              <div
                className={`rounded-2xl border bg-card transition-all ${expandedJob === j.id ? 'border-primary/50' : 'border-border hover:border-primary/30'}`}
              >
                {/* Job Header — always visible, click to expand */}
                <div
                  className="p-6 cursor-pointer"
                  onClick={() => setExpandedJob(expandedJob === j.id ? null : j.id)}
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className={`font-semibold transition-colors ${expandedJob === j.id ? 'text-primary' : 'text-foreground'}`}>
                        {j.title}
                      </h3>
                      <div className="flex items-center gap-4 mt-2">
                        <span className={`text-xs px-2 py-1 rounded-full ${teamColors[j.team] || ''}`}>{j.team}</span>
                        <span className="flex items-center gap-1 text-xs text-muted-foreground"><MapPin className="w-3 h-3" />{j.location}</span>
                        <span className="flex items-center gap-1 text-xs text-muted-foreground"><Clock className="w-3 h-3" />{j.type}</span>
                      </div>
                      <p className="text-sm text-muted-foreground mt-3">{j.desc}</p>
                    </div>
                    <div className="ml-4 flex-shrink-0 mt-1 text-muted-foreground transition-transform duration-200"
                      style={{ transform: expandedJob === j.id ? 'rotate(180deg)' : 'rotate(0deg)' }}>
                      <ChevronDown className="w-5 h-5" />
                    </div>
                  </div>
                </div>

                {/* Expanded job details */}
                <AnimatePresence>
                  {expandedJob === j.id && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.25 }}
                      className="overflow-hidden"
                    >
                      <div className="px-6 pb-6 border-t border-border pt-5">
                        <div className="prose prose-invert max-w-none">
                          {j.fullDesc.split('\n\n').map((block, idx) => (
                            <p key={idx} className="text-sm text-muted-foreground leading-relaxed mb-3 whitespace-pre-line">
                              {block}
                            </p>
                          ))}
                        </div>
                        <Button
                          className="mt-4"
                          onClick={() => setApplyJob(j)}
                        >
                          Apply for This Role
                        </Button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
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

      {/* Apply Modal */}
      <AnimatePresence>
        {applyJob && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm"
            onClick={() => setApplyJob(null)}
          >
            <motion.div
              initial={{ opacity: 0, y: 40, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 40, scale: 0.95 }}
              transition={{ type: 'spring', damping: 25 }}
              className="bg-card border border-border rounded-2xl max-w-md w-full shadow-2xl p-8"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold">Apply: {applyJob.title}</h2>
                <button onClick={() => setApplyJob(null)} className="text-muted-foreground hover:text-foreground p-1 rounded-lg hover:bg-muted">
                  <X className="w-5 h-5" />
                </button>
              </div>
              <p className="text-sm text-muted-foreground mb-6">
                Email your resume and portfolio to{' '}
                <a href={`mailto:careers@eyeassist.ai?subject=Application: ${applyJob.title}`} className="text-primary hover:underline">
                  careers@eyeassist.ai
                </a>{' '}
                with the subject line <span className="text-foreground font-medium">"{applyJob.title}"</span>.
              </p>
              <Button
                className="w-full"
                onClick={() => {
                  const a = document.createElement('a');
                  a.href = `mailto:careers@eyeassist.ai?subject=Application: ${applyJob.title}`;
                  a.target = '_blank';
                  a.rel = 'noopener noreferrer';
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                }}
              >
                <Mail className="w-4 h-4 mr-2" />
                Open Email Client
              </Button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
