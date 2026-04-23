import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
  Eye, Calendar, Clock, FileText, ChevronRight, CheckCircle, AlertCircle,
  BookOpen, Video, User, Phone, Mail, MapPin, Star, TrendingUp, Activity,
  Shield, ArrowLeft, Play, Download, Printer, X, ArrowRight,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import PaymentGateway from '@/components/PaymentGateway';

// ─── Auth helper ─────────────────────────────────────────────────────────────
function getSession() {
  try { return JSON.parse(localStorage.getItem('eye_session') || 'null'); } catch { return null; }
}

// ─── Static data ─────────────────────────────────────────────────────────────
const SCAN_HISTORY = [
  { id:'SCAN-001', date:'2026-03-20', type:'AI Screening', laterality:'OD', result:'Mild DR',      severity:1, confidence:87.5, doctor:'Dr. Sarah Mitchell', notes:'Early changes noted. Follow-up in 6 months.' },
  { id:'SCAN-002', date:'2026-02-15', type:'AI Screening', laterality:'OS', result:'Normal',        severity:0, confidence:94.2, doctor:'Dr. Sarah Mitchell', notes:'Healthy retina. Continue annual screening.' },
  { id:'SCAN-003', date:'2026-01-10', type:'AI Screening', laterality:'OD', result:'Moderate DR',   severity:2, confidence:91.8, doctor:'Dr. James Chen',     notes:'Progression noted. Referral to retina specialist.' },
  { id:'SCAN-004', date:'2025-12-05', type:'AI Screening', laterality:'OS', result:'Mild DR',      severity:1, confidence:82.3, doctor:'Dr. Sarah Mitchell', notes:'Stable condition. Monitor closely.' },
  { id:'SCAN-005', date:'2025-09-20', type:'AI Screening', laterality:'OD', result:'Normal',        severity:0, confidence:96.1, doctor:'Dr. Sarah Mitchell', notes:'Baseline scan. All clear.' },
];

const TREND_DATA = [
  { date:'Sep 25', severity:0, confidence:96 },
  { date:'Dec 25', severity:1, confidence:82 },
  { date:'Jan 26', severity:2, confidence:92 },
  { date:'Feb 26', severity:0, confidence:94 },
  { date:'Mar 26', severity:1, confidence:88 },
];

const DOCTORS = [
  { id:'DOC-001', name:'Dr. Sarah Mitchell', specialty:'Retina Specialist',       rating:4.9, reviews:128, initials:'SM', slots:['9:00 AM','11:30 AM','2:00 PM'],  fee:500 },
  { id:'DOC-002', name:'Dr. James Chen',     specialty:'Glaucoma Specialist',      rating:4.8, reviews:96,  initials:'JC', slots:['10:00 AM','1:30 PM','4:00 PM'], fee:600 },
  { id:'DOC-003', name:'Dr. Maria Garcia',   specialty:'General Ophthalmology',    rating:4.7, reviews:84,  initials:'MG', slots:['8:30 AM','12:00 PM','3:30 PM'], fee:400 },
];

const EDUCATIONAL_CONTENT = [
  { id:1, title:'Understanding Diabetic Retinopathy', type:'article', readTime:'5 min', category:'DR',         description:'Diabetic retinopathy (DR) is a diabetes complication that affects the eyes. It is caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina). DR progresses through four stages: mild, moderate, severe nonproliferative, and proliferative. Early detection through regular screening is the most effective way to prevent vision loss. Key warning signs include floaters, blurred vision, or sudden vision changes.' },
  { id:2, title:'How AI is Transforming Eye Care', type:'video', duration:'8:30',    category:'Technology',   youtubeId:'KBGWkRh_Z_Q', description:'Discover how artificial intelligence is helping detect eye diseases earlier than ever.' },
  { id:3, title:'Living with Diabetes: Eye Health Tips', type:'article', readTime:'7 min', category:'Prevention', description:'Managing your eyes when you have diabetes requires consistent effort. Monitor your blood sugar closely — HbA1c above 7% significantly increases DR risk. Get a dilated eye exam at least once a year. Control blood pressure (target <130/80 mmHg) and cholesterol. Quit smoking. Exercise regularly. Report any sudden vision changes immediately to your doctor.' },
  { id:4, title:'What Your Fundus Image Reveals', type:'video', duration:'5:45',    category:'Education',    youtubeId:'pxfBG7ORVOA', description:'A visual guide to understanding what doctors look for in retinal images.' },
];

const SEV: Record<number, { color:string; bg:string; border:string; label:string; desc:string }> = {
  0: { color:'text-emerald-400', bg:'bg-emerald-500/10', border:'border-emerald-500/30', label:'Normal',       desc:'No signs of disease' },
  1: { color:'text-amber-400',   bg:'bg-amber-500/10',   border:'border-amber-500/30',   label:'Mild',         desc:'Early signs, monitor' },
  2: { color:'text-orange-400',  bg:'bg-orange-500/10',  border:'border-orange-500/30',  label:'Moderate',     desc:'Follow-up needed' },
  3: { color:'text-red-400',     bg:'bg-red-500/10',     border:'border-red-500/30',     label:'Severe',       desc:'Urgent care required' },
  4: { color:'text-red-500',     bg:'bg-red-500/15',     border:'border-red-500/40',     label:'Proliferative',desc:'Immediate intervention' },
};

// ─── Main Component ───────────────────────────────────────────────────────────
export default function PatientPortal() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab]           = useState('reports');
  const [selectedScan, setSelectedScan]     = useState<string|null>(null);
  const [selectedDoctor, setSelectedDoctor] = useState<string|null>(null);
  const [selectedSlot, setSelectedSlot]     = useState<string|null>(null);
  const [bookingDone, setBookingDone]       = useState(false);
  const [videoModal, setVideoModal]         = useState<typeof EDUCATIONAL_CONTENT[0]|null>(null);
  const [articleModal, setArticleModal]     = useState<typeof EDUCATIONAL_CONTENT[0]|null>(null);
  const [showPayment, setShowPayment]       = useState(false);
  const [paymentCtx, setPaymentCtx]         = useState<{amount:number;desc:string}|null>(null);

  // Get real user from localStorage session
  const session = getSession();
  const userName    = session?.name  || 'Patient';
  const userEmail   = session?.email || '';
  const patientId   = session?.patientId || 'P-2026-0001';
  const nextAppt    = '2026-04-25';
  const primaryDoc  = 'Dr. Sarah Mitchell';

  const latestScan = SCAN_HISTORY[0];
  const sev        = SEV[latestScan.severity] || SEV[0];

  const triggerPayment = (amount: number, desc: string) => {
    setPaymentCtx({ amount, desc });
    setShowPayment(true);
  };

  const onPaymentSuccess = (_txnId: string) => {
    setShowPayment(false);
    setPaymentCtx(null);
    // If payment was for booking, confirm booking
    if (selectedDoctor && selectedSlot) {
      setBookingDone(true);
      setTimeout(() => { setBookingDone(false); setSelectedDoctor(null); setSelectedSlot(null); }, 5000);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="bg-card border-b border-border sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo + Back */}
            <div className="flex items-center gap-4">
              <Button variant="ghost" size="sm" onClick={() => navigate('/')} className="text-muted-foreground hover:text-foreground">
                <ArrowLeft className="w-4 h-4 mr-1" />
                <span className="hidden sm:inline">Back</span>
              </Button>
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
                  <Eye className="w-5 h-5 text-primary" />
                </div>
                <span className="font-bold text-foreground">EYE-ASSISST</span>
                <Badge variant="outline" className="text-xs hidden sm:inline-flex">Patient Portal</Badge>
              </div>
            </div>
            {/* User Info */}
            <div className="flex items-center gap-3">
              <div className="text-right hidden sm:block">
                <p className="text-sm font-medium text-foreground">{userName}</p>
                <p className="text-xs text-muted-foreground">{patientId}</p>
              </div>
              <div className="w-9 h-9 bg-primary/10 rounded-full flex items-center justify-center">
                <User className="w-5 h-5 text-primary" />
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* ── Welcome card ───────────────────────────────────────────── */}
        <motion.div initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} className="mb-8">
          <div className="bg-card rounded-2xl border border-border p-6">
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
              <div>
                <h1 className="text-2xl font-bold text-foreground mb-1">
                  Welcome back, {userName.split(' ')[0]} 👋
                </h1>
                <p className="text-muted-foreground">Here's an overview of your eye health</p>
              </div>
              <div className={`inline-flex items-center gap-3 px-5 py-3 rounded-xl border ${sev.bg} ${sev.border}`}>
                <Activity className={`w-6 h-6 ${sev.color}`} />
                <div>
                  <p className={`text-sm font-semibold ${sev.color}`}>Latest: {latestScan.result}</p>
                  <p className="text-xs text-muted-foreground">{latestScan.date} • {sev.desc}</p>
                </div>
              </div>
            </div>
            {/* Quick stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 pt-6 border-t border-border">
              <div className="text-center">
                <p className="text-2xl font-bold text-primary">{SCAN_HISTORY.length}</p>
                <p className="text-xs text-muted-foreground">Total Scans</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-primary">{nextAppt}</p>
                <p className="text-xs text-muted-foreground">Next Checkup</p>
              </div>
              <div className="text-center">
                <p className="text-lg font-bold text-primary">{primaryDoc}</p>
                <p className="text-xs text-muted-foreground">Primary Doctor</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-emerald-400">{latestScan.confidence}%</p>
                <p className="text-xs text-muted-foreground">AI Confidence</p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* ── Tabs ───────────────────────────────────────────────────── */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-card border border-border p-1 rounded-xl w-full sm:w-auto">
            {[
              { id:'reports',   icon:FileText,   label:'My Reports' },
              { id:'booking',   icon:Calendar,   label:'Book Appointment' },
              { id:'timeline',  icon:TrendingUp, label:'Health Timeline' },
              { id:'education', icon:BookOpen,   label:'Learn' },
            ].map(t => (
              <TabsTrigger
                key={t.id}
                value={t.id}
                className="rounded-lg px-4 py-2 data-[state=active]:bg-primary/10 data-[state=active]:text-primary"
              >
                <t.icon className="w-4 h-4 mr-1.5" />
                <span className="hidden sm:inline">{t.label}</span>
              </TabsTrigger>
            ))}
          </TabsList>

          {/* ── MY REPORTS ─────────────────────────────────────────── */}
          <TabsContent value="reports">
            <div className="grid lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 space-y-4">
                {SCAN_HISTORY.map((scan, i) => {
                  const s = SEV[scan.severity] || SEV[0];
                  return (
                    <motion.div key={scan.id} initial={{ opacity:0, x:-20 }} animate={{ opacity:1, x:0 }} transition={{ delay:i*0.07 }}>
                      <Card className={`cursor-pointer transition-all border-border bg-card hover:border-primary/30 ${selectedScan===scan.id ? 'ring-1 ring-primary' : ''}`}
                            onClick={() => setSelectedScan(selectedScan===scan.id ? null : scan.id)}>
                        <CardContent className="p-5">
                          <div className="flex items-start justify-between">
                            <div className="flex items-start gap-4">
                              <div className={`w-11 h-11 ${s.bg} ${s.border} border rounded-xl flex items-center justify-center`}>
                                <Eye className={`w-5 h-5 ${s.color}`} />
                              </div>
                              <div>
                                <div className="flex items-center gap-2">
                                  <h3 className="font-semibold text-foreground">{scan.type}</h3>
                                  <Badge variant="outline" className="text-xs">{scan.laterality==='OD' ? 'Right Eye':'Left Eye'}</Badge>
                                </div>
                                <p className="text-sm text-muted-foreground mt-1">{scan.date} • {scan.doctor}</p>
                                <div className="flex items-center gap-3 mt-2">
                                  <Badge className={`${s.bg} ${s.color} border-0 text-xs`}>{scan.result}</Badge>
                                  <span className="text-xs text-muted-foreground">Confidence: {scan.confidence}%</span>
                                </div>
                              </div>
                            </div>
                            <ChevronRight className={`w-5 h-5 text-muted-foreground transition-transform ${selectedScan===scan.id ? 'rotate-90':''}`} />
                          </div>

                          <AnimatePresence>
                            {selectedScan===scan.id && (
                              <motion.div initial={{ opacity:0, height:0 }} animate={{ opacity:1, height:'auto' }} exit={{ opacity:0, height:0 }}
                                          className="mt-4 pt-4 border-t border-border">
                                <div className="grid md:grid-cols-2 gap-4">
                                  <div>
                                    <p className="text-sm font-medium text-foreground mb-1">Doctor's Notes</p>
                                    <p className="text-sm text-muted-foreground bg-muted/30 p-3 rounded-lg">{scan.notes}</p>
                                  </div>
                                  <div>
                                    <p className="text-sm font-medium text-foreground mb-1">Analysis Details</p>
                                    <div className="space-y-2">
                                      <div className="flex justify-between text-sm">
                                        <span className="text-muted-foreground">Severity</span>
                                        <span className={`font-medium ${s.color}`}>{scan.severity}/4</span>
                                      </div>
                                      <Progress value={(scan.severity/4)*100} className="h-2" />
                                      <div className="flex justify-between text-sm">
                                        <span className="text-muted-foreground">AI Confidence</span>
                                        <span className="font-medium text-primary">{scan.confidence}%</span>
                                      </div>
                                      <Progress value={scan.confidence} className="h-2" />
                                    </div>
                                  </div>
                                </div>
                                <div className="flex gap-2 mt-4">
                                  <Button size="sm" variant="outline"><Download className="w-4 h-4 mr-1" />Download PDF</Button>
                                  <Button size="sm" variant="outline"><Printer className="w-4 h-4 mr-1" />Print</Button>
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </CardContent>
                      </Card>
                    </motion.div>
                  );
                })}
              </div>

              {/* Sidebar */}
              <div className="space-y-4">
                <Card className="bg-gradient-to-br from-cyan-500 to-blue-600 border-0 text-white">
                  <CardContent className="p-5">
                    <Shield className="w-7 h-7 mb-3 opacity-80" />
                    <h3 className="font-semibold text-lg mb-1">Your Data is Secure</h3>
                    <p className="text-sm opacity-90">Encrypted & stored in HIPAA compliance.</p>
                  </CardContent>
                </Card>

                <Card className="bg-card border-border">
                  <CardHeader className="pb-3"><CardTitle className="text-sm text-foreground">Quick Actions</CardTitle></CardHeader>
                  <CardContent className="space-y-2">
                    {[
                      { icon:Calendar, label:'Book Follow-up',    action: () => setActiveTab('booking') },
                      { icon:TrendingUp, label:'View Timeline',   action: () => setActiveTab('timeline') },
                      { icon:BookOpen, label:'Learn More',        action: () => setActiveTab('education') },
                    ].map(a => (
                      <Button key={a.label} variant="outline" className="w-full justify-start border-border" onClick={a.action}>
                        <a.icon className="w-4 h-4 mr-2" />{a.label}
                      </Button>
                    ))}
                  </CardContent>
                </Card>

                <Card className="bg-card border-border">
                  <CardHeader className="pb-3"><CardTitle className="text-sm text-foreground">Contact Info</CardTitle></CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    {[
                      { icon:User,  text:primaryDoc },
                      { icon:Phone, text:'+1 (555) 123-4567' },
                      { icon:Mail,  text:userEmail || 'patient@email.com' },
                    ].map(({ icon:Icon, text }) => (
                      <div key={text} className="flex items-center gap-2">
                        <Icon className="w-4 h-4 text-muted-foreground" />
                        <span className="text-muted-foreground">{text}</span>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* ── BOOK APPOINTMENT ───────────────────────────────────── */}
          <TabsContent value="booking">
            <AnimatePresence mode="wait">
              {bookingDone ? (
                <motion.div key="done" initial={{ opacity:0, scale:0.9 }} animate={{ opacity:1, scale:1 }}
                            className="max-w-lg mx-auto text-center py-16">
                  <div className="w-20 h-20 bg-emerald-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
                    <CheckCircle className="w-10 h-10 text-emerald-400" />
                  </div>
                  <h2 className="text-2xl font-bold text-foreground mb-2">Appointment Confirmed!</h2>
                  <p className="text-muted-foreground mb-2">
                    Your appointment with {DOCTORS.find(d=>d.id===selectedDoctor)?.name} is scheduled.
                  </p>
                  <p className="text-sm text-muted-foreground">A confirmation will be sent to your email.</p>
                </motion.div>
              ) : (
                <motion.div key="form" initial={{ opacity:0 }} animate={{ opacity:1 }}>
                  <div className="grid lg:grid-cols-3 gap-6">
                    {/* Doctor list */}
                    <div className="lg:col-span-2 space-y-4">
                      <h2 className="text-lg font-semibold text-foreground">Select a Doctor</h2>
                      {DOCTORS.map((doc, i) => (
                        <motion.div key={doc.id} initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} transition={{ delay:i*0.07 }}>
                          <Card className={`cursor-pointer transition-all border-border bg-card ${selectedDoctor===doc.id ? 'ring-1 ring-primary border-primary/30':'hover:border-primary/20'}`}
                                onClick={() => { setSelectedDoctor(doc.id); setSelectedSlot(null); }}>
                            <CardContent className="p-5">
                              <div className="flex items-center gap-4">
                                <div className="w-14 h-14 bg-primary/10 rounded-full flex items-center justify-center text-lg font-bold text-primary">
                                  {doc.initials}
                                </div>
                                <div className="flex-1">
                                  <div className="flex items-center gap-2 flex-wrap">
                                    <h3 className="font-semibold text-foreground">{doc.name}</h3>
                                    <div className="flex items-center gap-1">
                                      <Star className="w-3 h-3 text-amber-400 fill-amber-400" />
                                      <span className="text-sm text-foreground">{doc.rating}</span>
                                      <span className="text-xs text-muted-foreground">({doc.reviews})</span>
                                    </div>
                                  </div>
                                  <p className="text-sm text-muted-foreground">{doc.specialty}</p>
                                  <p className="text-sm text-primary font-medium mt-1">Consultation: ₹{doc.fee}</p>
                                </div>
                                <ChevronRight className={`w-5 h-5 text-muted-foreground transition-transform ${selectedDoctor===doc.id?'rotate-90':''}`} />
                              </div>

                              <AnimatePresence>
                                {selectedDoctor===doc.id && (
                                  <motion.div initial={{ opacity:0, height:0 }} animate={{ opacity:1, height:'auto' }} exit={{ opacity:0, height:0 }}
                                              className="mt-4 pt-4 border-t border-border">
                                    <p className="text-sm font-medium text-foreground mb-3">Available Tomorrow</p>
                                    <div className="flex flex-wrap gap-2">
                                      {doc.slots.map(slot => (
                                        <button key={slot} onClick={e => { e.stopPropagation(); setSelectedSlot(slot); }}
                                                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${selectedSlot===slot ? 'bg-primary text-primary-foreground':'bg-muted text-muted-foreground hover:bg-muted/70'}`}>
                                          {slot}
                                        </button>
                                      ))}
                                    </div>
                                  </motion.div>
                                )}
                              </AnimatePresence>
                            </CardContent>
                          </Card>
                        </motion.div>
                      ))}

                      {/* Confirm → triggers payment */}
                      <AnimatePresence>
                        {selectedDoctor && selectedSlot && (
                          <motion.div initial={{ opacity:0, y:10 }} animate={{ opacity:1, y:0 }} exit={{ opacity:0 }}>
                            <Button
                              size="lg"
                              className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700"
                              onClick={() => {
                                const doc = DOCTORS.find(d=>d.id===selectedDoctor);
                                triggerPayment(doc?.fee || 500, `Consultation with ${doc?.name} at ${selectedSlot}`);
                              }}
                            >
                              <Calendar className="w-5 h-5 mr-2" />
                              Pay & Confirm Appointment
                            </Button>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    {/* Info sidebar */}
                    <div className="space-y-4">
                      <Card className="bg-card border-border">
                        <CardHeader className="pb-3"><CardTitle className="text-sm text-foreground">Clinic Location</CardTitle></CardHeader>
                        <CardContent className="space-y-3">
                          {/* Live Google Maps embed */}
                          <iframe
                            title="Clinic Location"
                            src="https://maps.google.com/maps?q=Boston+Eye+Associates,Boston,MA&output=embed"
                            className="w-full h-44 rounded-xl border-0"
                            loading="lazy"
                          />
                          <div className="flex items-start gap-2 text-sm">
                            <MapPin className="w-4 h-4 text-muted-foreground mt-0.5 flex-shrink-0" />
                            <div>
                              <p className="font-medium text-foreground">Boston Eye Institute</p>
                              <p className="text-muted-foreground">123 Medical Center Dr, Boston, MA</p>
                            </div>
                          </div>
                          <div className="flex items-center gap-2 text-sm">
                            <Clock className="w-4 h-4 text-muted-foreground" />
                            <span className="text-muted-foreground">Duration: 30 minutes</span>
                          </div>
                          <div className="flex items-center gap-2 text-sm">
                            <Phone className="w-4 h-4 text-muted-foreground" />
                            <span className="text-muted-foreground">(555) 123-4567</span>
                          </div>
                        </CardContent>
                      </Card>

                      <Card className="bg-amber-500/10 border-amber-500/30">
                        <CardContent className="p-4">
                          <div className="flex items-start gap-2">
                            <AlertCircle className="w-5 h-5 text-amber-400 mt-0.5" />
                            <div>
                              <p className="font-medium text-amber-300 text-sm">Before Your Visit</p>
                              <ul className="text-xs text-muted-foreground mt-1 space-y-1">
                                <li>• Bring your insurance card</li>
                                <li>• List current medications</li>
                                <li>• Don't wear eye makeup</li>
                                <li>• Bring sunglasses (pupils may be dilated)</li>
                              </ul>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </TabsContent>

          {/* ── HEALTH TIMELINE ────────────────────────────────────── */}
          <TabsContent value="timeline">
            <div className="grid lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <Card className="bg-card border-border">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-foreground">
                      <TrendingUp className="w-5 h-5 text-primary" />Disease Severity Over Time
                    </CardTitle>
                    <CardDescription>Tracking your retinal health across all screenings</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-72">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={TREND_DATA}>
                          <defs>
                            <linearGradient id="sev" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%"  stopColor="#22D3EE" stopOpacity={0.3} />
                              <stop offset="95%" stopColor="#22D3EE" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                          <XAxis dataKey="date"     stroke="rgba(255,255,255,0.3)" fontSize={12} />
                          <YAxis stroke="rgba(255,255,255,0.3)" fontSize={12} domain={[0,4]} ticks={[0,1,2,3,4]} />
                          <Tooltip contentStyle={{ backgroundColor:'#111827', border:'1px solid rgba(255,255,255,0.1)', borderRadius:'8px', color:'#fff' }} />
                          <Area type="monotone" dataKey="severity" stroke="#22D3EE" fillOpacity={1} fill="url(#sev)" strokeWidth={2} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <div className="space-y-1">
                <h3 className="font-semibold text-foreground mb-3">Screening History</h3>
                {SCAN_HISTORY.map((scan, i) => {
                  const s = SEV[scan.severity] || SEV[0];
                  return (
                    <motion.div key={scan.id} initial={{ opacity:0, x:20 }} animate={{ opacity:1, x:0 }} transition={{ delay:i*0.07 }}
                                className="flex gap-3">
                      <div className="flex flex-col items-center">
                        <div className={`w-3 h-3 rounded-full mt-1.5 ${s.bg} border ${s.border}`} />
                        {i < SCAN_HISTORY.length-1 && <div className="w-px flex-1 bg-border mt-1" />}
                      </div>
                      <div className="pb-5">
                        <p className="text-sm font-medium text-foreground">{scan.date}</p>
                        <p className={`text-sm ${s.color}`}>{scan.result}</p>
                        <p className="text-xs text-muted-foreground">{scan.doctor}</p>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          </TabsContent>

          {/* ── EDUCATION ──────────────────────────────────────────── */}
          <TabsContent value="education">
            <div className="mb-5">
              <h2 className="text-lg font-semibold text-foreground mb-1">Educational Resources</h2>
              <p className="text-muted-foreground">Learn more about your condition and how to protect your vision</p>
            </div>
            <div className="grid md:grid-cols-2 gap-6">
              {EDUCATIONAL_CONTENT.map((c, i) => (
                <motion.div key={c.id} initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} transition={{ delay:i*0.08 }}>
                  <Card
                    className="overflow-hidden cursor-pointer group transition-all hover:border-primary/30 bg-card border-border"
                    onClick={() => c.type==='video' ? setVideoModal(c) : setArticleModal(c)}
                  >
                    {/* Thumbnail */}
                    <div className="h-40 bg-primary/5 flex items-center justify-center relative overflow-hidden">
                      {c.type==='video' ? (
                        <>
                          <Video className="w-12 h-12 text-primary/40" />
                          <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/40">
                            <div className="w-14 h-14 bg-primary/90 rounded-full flex items-center justify-center">
                              <Play className="w-6 h-6 text-white ml-1" />
                            </div>
                          </div>
                          <div className="absolute bottom-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                            {c.duration}
                          </div>
                        </>
                      ) : (
                        <BookOpen className="w-12 h-12 text-primary/40" />
                      )}
                      <Badge className="absolute top-2 left-2 bg-card/90 text-foreground border-border text-xs">
                        {c.category}
                      </Badge>
                    </div>
                    <CardContent className="p-5">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="outline" className="text-xs">{c.type}</Badge>
                        <span className="text-xs text-muted-foreground">{c.readTime || c.duration}</span>
                      </div>
                      <h3 className="font-semibold text-foreground mb-1">{c.title}</h3>
                      <p className="text-sm text-muted-foreground line-clamp-2">{c.description}</p>
                      <div className="flex items-center gap-1 mt-3 text-primary text-sm font-medium">
                        {c.type==='video' ? 'Watch now' : 'Read more'}
                        <ArrowRight className="w-4 h-4" />
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>

      {/* ── Video Modal ──────────────────────────────────────────────── */}
      <AnimatePresence>
        {videoModal && (
          <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0 }}
                      className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4"
                      onClick={() => setVideoModal(null)}>
            <motion.div initial={{ scale:0.9 }} animate={{ scale:1 }} exit={{ scale:0.9 }}
                        className="w-full max-w-3xl" onClick={e => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold text-white">{videoModal.title}</h3>
                <Button variant="ghost" size="icon" className="text-white" onClick={() => setVideoModal(null)}>
                  <X className="w-5 h-5" />
                </Button>
              </div>
              <iframe
                src={`https://www.youtube.com/embed/${videoModal.youtubeId}?autoplay=1`}
                className="w-full aspect-video rounded-2xl"
                allowFullScreen
                allow="autoplay; encrypted-media"
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Article Modal ─────────────────────────────────────────────── */}
      <AnimatePresence>
        {articleModal && (
          <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0 }}
                      className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
                      onClick={() => setArticleModal(null)}>
            <motion.div initial={{ y:30, opacity:0 }} animate={{ y:0, opacity:1 }} exit={{ y:30, opacity:0 }}
                        className="bg-card border border-border rounded-2xl p-6 max-w-lg w-full" onClick={e => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-4">
                <Badge variant="outline">{articleModal.category}</Badge>
                <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setArticleModal(null)}>
                  <X className="w-4 h-4" />
                </Button>
              </div>
              <h3 className="text-xl font-bold text-foreground mb-3">{articleModal.title}</h3>
              <p className="text-muted-foreground leading-relaxed">{articleModal.description}</p>
              <Button className="w-full mt-6" onClick={() => setArticleModal(null)}>Got it</Button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Payment Gateway ───────────────────────────────────────────── */}
      <AnimatePresence>
        {showPayment && paymentCtx && (
          <PaymentGateway
            amount={paymentCtx.amount}
            description={paymentCtx.desc}
            onSuccess={onPaymentSuccess}
            onClose={() => { setShowPayment(false); setPaymentCtx(null); }}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
