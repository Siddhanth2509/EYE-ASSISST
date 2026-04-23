import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, FileText, TrendingUp, Users, Award } from 'lucide-react';
import { Button } from '@/components/ui/button';

const cases = [
  {
    hospital:'Apollo Hospitals — Mumbai',
    disease:'Diabetic Retinopathy',
    stat:'40% faster screening',
    patients:1200,
    highlight:'EYE-ASSISST reduced average report turnaround from 48 hours to under 5 minutes, enabling same-day DR grading for 1,200+ outpatients.',
    quote:'"The Grad-CAM explanations gave our junior residents immediate confidence in the AI findings." — Dr. R. Mehta, Retina Specialist',
    color:'from-cyan-500 to-blue-600',
  },
  {
    hospital:'Sankara Nethralaya — Chennai',
    disease:'Glaucoma Screening',
    stat:'94% sensitivity',
    patients:850,
    highlight:'Deployed for high-volume glaucoma screening in a resource-constrained setting. The AI flagged 97 previously undiagnosed cases in the first quarter.',
    quote:'"We screened twice as many patients without increasing ophthalmologist workload." — Dr. S. Krishnan, HOD Glaucoma',
    color:'from-purple-500 to-indigo-600',
  },
  {
    hospital:'Aravind Eye Hospital — Madurai',
    disease:'Multi-disease Rural Screening',
    stat:'3x throughput',
    patients:3400,
    highlight:'Used during rural telemedicine camps. A single camera and tablet enabled 3,400 screenings in 6 districts — results analyzed by AI and reviewed by remote ophthalmologists.',
    quote:'"AI screening changed what\'s possible in underserved communities." — Dr. P. Venkataswamy',
    color:'from-emerald-500 to-teal-600',
  },
];

export default function CaseStudiesPage() {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <Button variant="ghost" size="sm" onClick={() => navigate('/')} className="mb-8 text-muted-foreground">
          <ArrowLeft className="w-4 h-4 mr-2" />Back to Home
        </Button>

        <motion.div initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} className="mb-12">
          <div className="flex items-center gap-2 mb-3">
            <FileText className="w-5 h-5 text-primary" />
            <span className="text-sm text-primary font-medium">Case Studies</span>
          </div>
          <h1 className="text-4xl font-bold mb-3">Real Impact, <span className="text-primary">Real Patients</span></h1>
          <p className="text-muted-foreground text-lg max-w-2xl">
            How leading eye hospitals across India are using EYE-ASSISST to transform clinical workflows.
          </p>
        </motion.div>

        <div className="space-y-8">
          {cases.map((c,i) => (
            <motion.div key={c.hospital} initial={{ opacity:0, y:30 }} animate={{ opacity:1, y:0 }} transition={{ delay:i*0.1 }}
              className="rounded-2xl border border-border bg-card overflow-hidden">
              <div className={`h-2 bg-gradient-to-r ${c.color}`} />
              <div className="p-7">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-5">
                  <div>
                    <h2 className="text-xl font-bold text-foreground">{c.hospital}</h2>
                    <p className="text-sm text-muted-foreground">{c.disease}</p>
                  </div>
                  <div className="flex gap-5">
                    <div className="text-center">
                      <p className={`text-2xl font-bold bg-gradient-to-r ${c.color} bg-clip-text text-transparent`}>{c.stat}</p>
                      <p className="text-xs text-muted-foreground">Improvement</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-foreground">{c.patients.toLocaleString()}</p>
                      <p className="text-xs text-muted-foreground">Patients</p>
                    </div>
                  </div>
                </div>
                <p className="text-sm text-muted-foreground mb-4 leading-relaxed">{c.highlight}</p>
                <blockquote className="border-l-2 border-primary/40 pl-4 text-sm text-muted-foreground italic">
                  {c.quote}
                </blockquote>
              </div>
            </motion.div>
          ))}
        </div>

        <div className="grid sm:grid-cols-3 gap-4 mt-12">
          {[
            { icon:Users,    label:'Patients Screened',  value:'5,450+' },
            { icon:Award,    label:'Partner Hospitals',  value:'3' },
            { icon:TrendingUp, label:'Avg. Accuracy',   value:'94%' },
          ].map(s => (
            <div key={s.label} className="p-5 rounded-2xl border border-border bg-card text-center">
              <s.icon className="w-6 h-6 text-primary mx-auto mb-2" />
              <p className="text-2xl font-bold text-foreground">{s.value}</p>
              <p className="text-xs text-muted-foreground mt-1">{s.label}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
