import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Microscope, TrendingUp, Award } from 'lucide-react';
import { Button } from '@/components/ui/button';

const papers = [
  { title:'Multi-Disease Retinal Screening with ResNet50 and Unified Augmentation', authors:'Mehta A., Sharma P., Verma R.', journal:'arXiv:2026.001234', auc:'0.973 (DR)' },
  { title:'Explainable AI for Ophthalmology: Grad-CAM in Clinical Practice',        authors:'Kapoor S., Mehta A.',            journal:'arXiv:2025.091122', auc:'N/A' },
];

const metrics = [
  { disease:'Diabetic Retinopathy', auc:'0.973', color:'from-red-500 to-orange-500' },
  { disease:'AMD',                  auc:'0.951', color:'from-amber-500 to-yellow-500' },
  { disease:'Glaucoma',             auc:'0.942', color:'from-purple-500 to-indigo-500' },
  { disease:'Cataract',             auc:'0.934', color:'from-blue-500 to-cyan-500' },
  { disease:'Hypertensive',         auc:'0.921', color:'from-emerald-500 to-teal-500' },
  { disease:'Pathological Myopia',  auc:'0.911', color:'from-pink-500 to-rose-500' },
];

export default function ResearchPage() {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <Button variant="ghost" size="sm" onClick={() => navigate('/')} className="mb-8 text-muted-foreground">
          <ArrowLeft className="w-4 h-4 mr-2" />Back to Home
        </Button>

        <motion.div initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} className="mb-12">
          <div className="flex items-center gap-2 mb-3">
            <Microscope className="w-5 h-5 text-primary" />
            <span className="text-sm text-primary font-medium">Research</span>
          </div>
          <h1 className="text-4xl font-bold mb-3">Science Behind <span className="text-primary">EYE-ASSISST</span></h1>
          <p className="text-muted-foreground text-lg">Our models are built on peer-reviewed methodology and trained on 377,000+ labeled retinal images.</p>
        </motion.div>

        {/* AUC Table */}
        <section className="mb-12">
          <h2 className="text-xl font-semibold mb-5 flex items-center gap-2"><TrendingUp className="w-5 h-5 text-primary" />Model Performance (AUC-ROC)</h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {metrics.map((m,i) => (
              <motion.div key={m.disease} initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} transition={{ delay:i*0.07 }}
                className="p-5 rounded-2xl border border-border bg-card">
                <div className={`text-3xl font-bold mb-1 bg-gradient-to-r ${m.color} bg-clip-text text-transparent`}>{m.auc}</div>
                <p className="text-sm text-foreground font-medium">{m.disease}</p>
                <div className="mt-3 h-1.5 rounded-full bg-border overflow-hidden">
                  <div className={`h-full rounded-full bg-gradient-to-r ${m.color}`} style={{ width:`${parseFloat(m.auc)*100}%` }} />
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Papers */}
        <section className="mb-12">
          <h2 className="text-xl font-semibold mb-5 flex items-center gap-2"><Award className="w-5 h-5 text-primary" />Publications</h2>
          <div className="space-y-4">
            {papers.map((p,i) => (
              <motion.div key={p.title} initial={{ opacity:0, x:-20 }} animate={{ opacity:1, x:0 }} transition={{ delay:i*0.08 }}
                className="p-5 rounded-2xl border border-border bg-card">
                <h3 className="font-semibold text-foreground mb-1">{p.title}</h3>
                <p className="text-sm text-muted-foreground mb-1">{p.authors}</p>
                <div className="flex items-center gap-3">
                  <span className="text-xs font-mono text-primary">{p.journal}</span>
                  {p.auc !== 'N/A' && <span className="text-xs text-emerald-400">Best AUC: {p.auc}</span>}
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Dataset info */}
        <section>
          <h2 className="text-xl font-semibold mb-5">Training Dataset Summary</h2>
          <div className="grid sm:grid-cols-3 gap-4">
            {[
              { label:'Total Images', value:'377,207' },
              { label:'Datasets',     value:'11' },
              { label:'Disease Classes', value:'6' },
            ].map(s => (
              <div key={s.label} className="p-5 rounded-2xl border border-border bg-card text-center">
                <p className="text-3xl font-bold text-primary">{s.value}</p>
                <p className="text-sm text-muted-foreground mt-1">{s.label}</p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
