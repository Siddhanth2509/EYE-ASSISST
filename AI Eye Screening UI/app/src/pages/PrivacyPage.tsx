import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Shield, Lock, Eye, Database } from 'lucide-react';
import { Button } from '@/components/ui/button';

const sections = [
  {
    icon:Lock,
    title:'Data Collection & Use',
    text:`When you use EYE-ASSISST, we collect:
• Retinal fundus images submitted for analysis (processed, not stored permanently unless you opt in)
• Patient identifiers provided at scan time (Patient ID, laterality)
• Account information (email address, name, role)
• Usage analytics (anonymized, no PII)

We use your data solely to provide AI diagnostic services and improve our models. We do NOT sell your data to third parties.`,
  },
  {
    icon:Shield,
    title:'HIPAA Compliance',
    text:`EYE-ASSISST is designed with HIPAA compliance in mind:
• All data is encrypted in transit (TLS 1.3) and at rest (AES-256)
• Access is role-based — only authorized personnel can view patient records
• Audit logs track every data access event
• Business Associate Agreements (BAAs) are available for enterprise customers`,
  },
  {
    icon:Database,
    title:'Data Retention',
    text:`• Scan results are retained for 7 years (standard medical record retention)
• Account data is retained until you request deletion
• Anonymized model training data may be retained indefinitely
• You may request deletion of your personal data at any time by contacting privacy@eyeassist.ai`,
  },
  {
    icon:Eye,
    title:'Your Rights',
    text:`Under applicable data protection laws (DPDPA 2023, GDPR for EU users), you have the right to:
• Access your personal data
• Correct inaccurate data
• Request deletion ("right to be forgotten")
• Data portability (export your records in JSON/PDF)
• Withdraw consent at any time

To exercise any of these rights, contact: privacy@eyeassist.ai`,
  },
];

export default function PrivacyPage() {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <Button variant="ghost" size="sm" onClick={() => navigate('/')} className="mb-8 text-muted-foreground">
          <ArrowLeft className="w-4 h-4 mr-2" />Back to Home
        </Button>

        <motion.div initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} className="mb-12">
          <div className="flex items-center gap-2 mb-3">
            <Shield className="w-5 h-5 text-primary" />
            <span className="text-sm text-primary font-medium">Privacy Policy</span>
          </div>
          <h1 className="text-4xl font-bold mb-3">Your <span className="text-primary">Privacy</span> Matters</h1>
          <p className="text-muted-foreground">Last updated: April 18, 2026 · EYE-ASSISST Technologies Pvt. Ltd.</p>
        </motion.div>

        <div className="space-y-6">
          {sections.map((s,i) => (
            <motion.div key={s.title} initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} transition={{ delay:i*0.08 }}
              className="p-6 rounded-2xl border border-border bg-card">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-9 h-9 bg-primary/10 rounded-lg flex items-center justify-center">
                  <s.icon className="w-5 h-5 text-primary" />
                </div>
                <h2 className="text-lg font-semibold text-foreground">{s.title}</h2>
              </div>
              <pre className="text-sm text-muted-foreground whitespace-pre-wrap leading-relaxed font-sans">{s.text}</pre>
            </motion.div>
          ))}
        </div>

        <div className="mt-8 p-5 rounded-2xl bg-muted/30 border border-border text-center">
          <p className="text-sm text-muted-foreground">Questions about this policy? <button onClick={() => navigate('/contact')} className="text-primary hover:underline">Contact our Privacy team</button></p>
        </div>
      </div>
    </div>
  );
}
