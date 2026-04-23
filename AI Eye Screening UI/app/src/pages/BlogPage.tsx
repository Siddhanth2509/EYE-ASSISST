import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, BookOpen, Clock, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

const posts = [
  {
    id:1,
    title:'How We Achieved 0.97 AUC on Diabetic Retinopathy Detection',
    category:'Research',
    date:'April 10, 2026',
    readTime:'8 min',
    excerpt:'A deep dive into our training pipeline, dataset curation, and the architectural choices that pushed our DR model to clinical-grade accuracy.',
  },
  {
    id:2,
    title:'Why Grad-CAM Explainability Matters in Medical AI',
    category:'Technology',
    date:'March 28, 2026',
    readTime:'6 min',
    excerpt:'When AI makes a medical decision, doctors need to understand why. We explain how Grad-CAM heatmaps build trust and improve clinical workflows.',
  },
  {
    id:3,
    title:'Building a Multi-Disease Retinal Screening Platform from Scratch',
    category:'Engineering',
    date:'March 15, 2026',
    readTime:'12 min',
    excerpt:'How we consolidated 11 datasets, 377,000 images, and 6 disease classes into a single unified training pipeline using PyTorch and FastAPI.',
  },
  {
    id:4,
    title:'The State of AI in Ophthalmology: 2026 Review',
    category:'Industry',
    date:'March 1, 2026',
    readTime:'10 min',
    excerpt:'A comprehensive review of the latest AI models, clinical trials, and regulatory approvals shaping retinal disease detection globally.',
  },
];

const catColors: Record<string,string> = {
  Research:'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
  Technology:'bg-purple-500/10 text-purple-400 border-purple-500/20',
  Engineering:'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  Industry:'bg-amber-500/10 text-amber-400 border-amber-500/20',
};

export default function BlogPage() {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <Button variant="ghost" size="sm" onClick={() => navigate('/')} className="mb-8 text-muted-foreground">
          <ArrowLeft className="w-4 h-4 mr-2" />Back to Home
        </Button>

        <motion.div initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} className="mb-12">
          <div className="flex items-center gap-2 mb-3">
            <BookOpen className="w-5 h-5 text-primary" />
            <span className="text-sm text-primary font-medium">Blog</span>
          </div>
          <h1 className="text-4xl font-bold mb-3">Insights & <span className="text-primary">Updates</span></h1>
          <p className="text-muted-foreground text-lg">Latest thinking from the EYE-ASSISST research and engineering team.</p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-6">
          {posts.map((p,i) => (
            <motion.div key={p.id} initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} transition={{ delay:i*0.08 }}>
              <div className="p-6 rounded-2xl border border-border bg-card hover:border-primary/30 transition-colors group cursor-pointer h-full flex flex-col">
                <div className="flex items-center gap-3 mb-4">
                  <span className={`text-xs px-3 py-1 rounded-full border ${catColors[p.category]||''}`}>{p.category}</span>
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Clock className="w-3 h-3" />{p.readTime}
                  </div>
                </div>
                <h2 className="text-lg font-semibold text-foreground mb-2 group-hover:text-primary transition-colors">{p.title}</h2>
                <p className="text-sm text-muted-foreground flex-1 leading-relaxed">{p.excerpt}</p>
                <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
                  <span className="text-xs text-muted-foreground">{p.date}</span>
                  <span className="text-xs text-primary flex items-center gap-1">Read more <ChevronRight className="w-3 h-3" /></span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
