import { useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Send, Mail, Phone, MapPin, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';

export default function ContactPage() {
  const navigate = useNavigate();
  const [sent, setSent] = useState(false);
  const [form, setForm] = useState({ name:'', email:'', subject:'', message:'' });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSent(true);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <Button variant="ghost" size="sm" onClick={() => navigate('/')} className="mb-8 text-muted-foreground">
          <ArrowLeft className="w-4 h-4 mr-2" />Back to Home
        </Button>

        <motion.div initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }}>
          <h1 className="text-4xl font-bold mb-3">Get in <span className="text-primary">Touch</span></h1>
          <p className="text-muted-foreground text-lg mb-12">Have a question or want to partner with us? We'd love to hear from you.</p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-10">
          {/* Contact info */}
          <motion.div initial={{ opacity:0, x:-20 }} animate={{ opacity:1, x:0 }} className="space-y-6">
            {[
              { icon:Mail,    label:'Email',    value:'kuldeepgoswami636@gmail.com' },
              { icon:Phone,   label:'Phone',    value:'+91 7078860629' },
              { icon:MapPin,  label:'Address',  value:'5 Km Stone, Delhi-Meerut Road, Opposite Jain Tube Co. Ltd., Ghaziabad, Uttar Pradesh 201001, India' },
            ].map(({ icon:Icon, label, value }) => (
              <div key={label} className="flex items-center gap-4 p-5 rounded-2xl border border-border bg-card">
                <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center flex-shrink-0">
                  <Icon className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">{label}</p>
                  <p className="font-medium text-foreground">{value}</p>
                </div>
              </div>
            ))}

            <div className="p-5 rounded-2xl bg-primary/5 border border-primary/20">
              <p className="text-sm text-foreground font-medium mb-1">Sales Enquiries</p>
              <p className="text-sm text-muted-foreground">For Enterprise pricing or partnership discussions, email us at <span className="text-primary">sales@eyeassist.ai</span></p>
            </div>
          </motion.div>

          {/* Form */}
          <motion.div initial={{ opacity:0, x:20 }} animate={{ opacity:1, x:0 }}>
            {sent ? (
              <div className="text-center py-16">
                <CheckCircle className="w-16 h-16 text-emerald-400 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-foreground mb-2">Message Sent!</h2>
                <p className="text-muted-foreground">We'll get back to you within 24 hours.</p>
                <Button className="mt-6" onClick={() => setSent(false)}>Send Another</Button>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-4 bg-card border border-border rounded-2xl p-6">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Name</Label>
                    <Input placeholder="Your name" value={form.name} onChange={e=>setForm({...form,name:e.target.value})} required />
                  </div>
                  <div className="space-y-2">
                    <Label>Email</Label>
                    <Input type="email" placeholder="your@email.com" value={form.email} onChange={e=>setForm({...form,email:e.target.value})} required />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Subject</Label>
                  <Input placeholder="How can we help?" value={form.subject} onChange={e=>setForm({...form,subject:e.target.value})} required />
                </div>
                <div className="space-y-2">
                  <Label>Message</Label>
                  <Textarea placeholder="Tell us more..." className="min-h-[140px]" value={form.message} onChange={e=>setForm({...form,message:e.target.value})} required />
                </div>
                <Button type="submit" className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700">
                  <Send className="w-4 h-4 mr-2" />Send Message
                </Button>
              </form>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
}
