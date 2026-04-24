import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Eye,
  Calendar,
  Clock,
  FileText,
  ChevronRight,
  CheckCircle,
  AlertCircle,
  BookOpen,
  Video,
  User,
  Phone,
  Mail,
  MapPin,
  Star,
  TrendingUp,
  Activity,
  Shield,
  Sparkles,
  ArrowRight,
  Play,
  Download,
  Printer,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

// Mock patient data
const PATIENT_DATA = {
  id: 'P-2026-0041',
  name: 'John Anderson',
  age: 58,
  gender: 'Male',
  email: 'john.anderson@email.com',
  phone: '+1 (555) 123-4567',
  address: '123 Main St, Boston, MA 02101',
  insurance: 'Blue Cross Blue Shield',
  memberSince: '2023-01-15',
  nextAppointment: '2026-04-25',
  primaryDoctor: 'Dr. Sarah Mitchell',
};

// Mock scan history
const SCAN_HISTORY = [
  {
    id: 'SCAN-001',
    date: '2026-03-20',
    type: 'AI Screening',
    laterality: 'OD',
    result: 'Mild DR',
    severity: 1,
    confidence: 87.5,
    status: 'reviewed',
    doctor: 'Dr. Sarah Mitchell',
    notes: 'Early changes noted. Follow-up in 6 months.',
  },
  {
    id: 'SCAN-002',
    date: '2026-02-15',
    type: 'AI Screening',
    laterality: 'OS',
    result: 'Normal',
    severity: 0,
    confidence: 94.2,
    status: 'reviewed',
    doctor: 'Dr. Sarah Mitchell',
    notes: 'Healthy retina. Continue annual screening.',
  },
  {
    id: 'SCAN-003',
    date: '2026-01-10',
    type: 'AI Screening',
    laterality: 'OD',
    result: 'Moderate DR',
    severity: 2,
    confidence: 91.8,
    status: 'reviewed',
    doctor: 'Dr. James Chen',
    notes: 'Progression noted. Referral to retina specialist.',
  },
  {
    id: 'SCAN-004',
    date: '2025-12-05',
    type: 'AI Screening',
    laterality: 'OS',
    result: 'Mild DR',
    severity: 1,
    confidence: 82.3,
    status: 'reviewed',
    doctor: 'Dr. Sarah Mitchell',
    notes: 'Stable condition. Monitor closely.',
  },
  {
    id: 'SCAN-005',
    date: '2025-09-20',
    type: 'AI Screening',
    laterality: 'OD',
    result: 'Normal',
    severity: 0,
    confidence: 96.1,
    status: 'reviewed',
    doctor: 'Dr. Sarah Mitchell',
    notes: 'Baseline scan. All clear.',
  },
];

// Trend data for chart
const TREND_DATA = [
  { date: 'Sep 2025', severity: 0, confidence: 96 },
  { date: 'Dec 2025', severity: 1, confidence: 82 },
  { date: 'Jan 2026', severity: 2, confidence: 92 },
  { date: 'Feb 2026', severity: 0, confidence: 94 },
  { date: 'Mar 2026', severity: 1, confidence: 88 },
];

// Doctor availability
const DOCTORS = [
  {
    id: 'DOC-001',
    name: 'Dr. Sarah Mitchell',
    specialty: 'Retina Specialist',
    rating: 4.9,
    reviews: 128,
    image: 'SM',
    availableSlots: ['9:00 AM', '11:30 AM', '2:00 PM'],
  },
  {
    id: 'DOC-002',
    name: 'Dr. James Chen',
    specialty: 'Glaucoma Specialist',
    rating: 4.8,
    reviews: 96,
    image: 'JC',
    availableSlots: ['10:00 AM', '1:30 PM', '4:00 PM'],
  },
  {
    id: 'DOC-003',
    name: 'Dr. Maria Garcia',
    specialty: 'General Ophthalmology',
    rating: 4.7,
    reviews: 84,
    image: 'MG',
    availableSlots: ['8:30 AM', '12:00 PM', '3:30 PM'],
  },
];

// Educational content
const EDUCATIONAL_CONTENT = [
  {
    id: 1,
    title: 'Understanding Diabetic Retinopathy',
    type: 'article',
    readTime: '5 min',
    category: 'DR',
    description: 'Learn about the stages, symptoms, and treatment options for diabetic retinopathy.',
  },
  {
    id: 2,
    title: 'How AI is Transforming Eye Care',
    type: 'video',
    duration: '8:30',
    category: 'Technology',
    description: 'Discover how artificial intelligence is helping detect eye diseases earlier than ever.',
  },
  {
    id: 3,
    title: 'Living with Diabetes: Eye Health Tips',
    type: 'article',
    readTime: '7 min',
    category: 'Prevention',
    description: 'Practical lifestyle tips to protect your vision when you have diabetes.',
  },
  {
    id: 4,
    title: 'What Your Fundus Image Reveals',
    type: 'video',
    duration: '5:45',
    category: 'Education',
    description: 'A visual guide to understanding what doctors look for in retinal images.',
  },
];

// Severity config
const SEVERITY_CONFIG: Record<number, { color: string; bg: string; label: string; description: string }> = {
  0: { color: 'text-emerald-600', bg: 'bg-emerald-50', label: 'Normal', description: 'No signs of disease detected' },
  1: { color: 'text-amber-600', bg: 'bg-amber-50', label: 'Mild', description: 'Early signs, monitoring recommended' },
  2: { color: 'text-orange-600', bg: 'bg-orange-50', label: 'Moderate', description: 'Significant changes, follow-up needed' },
  3: { color: 'text-red-600', bg: 'bg-red-50', label: 'Severe', description: 'Advanced disease, urgent care required' },
  4: { color: 'text-red-700', bg: 'bg-red-100', label: 'Proliferative', description: 'Critical, immediate intervention needed' },
};

export default function PatientPortal() {
  const [activeTab, setActiveTab] = useState('reports');
  const [selectedDoctor, setSelectedDoctor] = useState<string | null>(null);
  const [selectedSlot, setSelectedSlot] = useState<string | null>(null);
  const [bookingConfirmed, setBookingConfirmed] = useState(false);
  const [selectedScan, setSelectedScan] = useState<string | null>(null);

  const handleBookAppointment = () => {
    if (!selectedDoctor || !selectedSlot) return;
    setBookingConfirmed(true);
    setTimeout(() => {
      setBookingConfirmed(false);
      setSelectedDoctor(null);
      setSelectedSlot(null);
    }, 3000);
  };

  const latestScan = SCAN_HISTORY[0];
  const severityConfig = SEVERITY_CONFIG[latestScan.severity] || SEVERITY_CONFIG[0];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50/30">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center">
                <Eye className="w-5 h-5 text-white" />
              </div>
              <div>
                <span className="font-bold text-gray-900">EYE-ASSISST</span>
                <span className="text-xs text-gray-500 ml-2">Patient Portal</span>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right hidden sm:block">
                <p className="text-sm font-medium text-gray-900">{PATIENT_DATA.name}</p>
                <p className="text-xs text-gray-500">{PATIENT_DATA.id}</p>
              </div>
              <div className="w-10 h-10 bg-gradient-to-br from-cyan-100 to-blue-100 rounded-full flex items-center justify-center">
                <User className="w-5 h-5 text-cyan-700" />
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome + Status Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
              <div>
                <h1 className="text-2xl font-bold text-gray-900 mb-1">
                  Welcome back, {PATIENT_DATA.name.split(' ')[0]}
                </h1>
                <p className="text-gray-500">Here's an overview of your eye health</p>
              </div>

              {/* Latest Result Summary */}
              <div className={`inline-flex items-center gap-3 px-5 py-3 rounded-xl ${severityConfig.bg}`}>
                <div className={`w-12 h-12 rounded-full flex items-center justify-center bg-white`}>
                  <Activity className={`w-6 h-6 ${severityConfig.color}`} />
                </div>
                <div>
                  <p className={`text-sm font-semibold ${severityConfig.color}`}>
                    Latest: {latestScan.result}
                  </p>
                  <p className="text-xs text-gray-600">
                    {latestScan.date} • {severityConfig.description}
                  </p>
                </div>
              </div>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 pt-6 border-t border-slate-100">
              <div className="text-center">
                <p className="text-2xl font-bold text-cyan-600">{SCAN_HISTORY.length}</p>
                <p className="text-xs text-gray-500">Total Scans</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-cyan-600">{PATIENT_DATA.nextAppointment}</p>
                <p className="text-xs text-gray-500">Next Checkup</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-cyan-600">{PATIENT_DATA.primaryDoctor}</p>
                <p className="text-xs text-gray-500">Primary Doctor</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-emerald-600">{latestScan.confidence}%</p>
                <p className="text-xs text-gray-500">AI Confidence</p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Main Content Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-white border border-slate-200 p-1 rounded-xl">
            <TabsTrigger value="reports" className="rounded-lg px-6 py-2.5 data-[state=active]:bg-cyan-50 data-[state=active]:text-cyan-700">
              <FileText className="w-4 h-4 mr-2" />
              My Reports
            </TabsTrigger>
            <TabsTrigger value="booking" className="rounded-lg px-6 py-2.5 data-[state=active]:bg-cyan-50 data-[state=active]:text-cyan-700">
              <Calendar className="w-4 h-4 mr-2" />
              Book Appointment
            </TabsTrigger>
            <TabsTrigger value="timeline" className="rounded-lg px-6 py-2.5 data-[state=active]:bg-cyan-50 data-[state=active]:text-cyan-700">
              <TrendingUp className="w-4 h-4 mr-2" />
              Health Timeline
            </TabsTrigger>
            <TabsTrigger value="education" className="rounded-lg px-6 py-2.5 data-[state=active]:bg-cyan-50 data-[state=active]:text-cyan-700">
              <BookOpen className="w-4 h-4 mr-2" />
              Learn
            </TabsTrigger>
          </TabsList>

          {/* My Reports Tab */}
          <TabsContent value="reports">
            <div className="grid lg:grid-cols-3 gap-6">
              {/* Scan List */}
              <div className="lg:col-span-2 space-y-4">
                {SCAN_HISTORY.map((scan, index) => {
                  const config = SEVERITY_CONFIG[scan.severity] || SEVERITY_CONFIG[0];
                  return (
                    <motion.div
                      key={scan.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                    >
                      <Card
                        className={`cursor-pointer transition-all hover:shadow-md ${
                          selectedScan === scan.id ? 'ring-2 ring-cyan-500' : ''
                        }`}
                        onClick={() => setSelectedScan(selectedScan === scan.id ? null : scan.id)}
                      >
                        <CardContent className="p-5">
                          <div className="flex items-start justify-between">
                            <div className="flex items-start gap-4">
                              <div className={`w-12 h-12 ${config.bg} rounded-xl flex items-center justify-center`}>
                                <Eye className={`w-6 h-6 ${config.color}`} />
                              </div>
                              <div>
                                <div className="flex items-center gap-2">
                                  <h3 className="font-semibold text-gray-900">{scan.type}</h3>
                                  <Badge variant="outline" className="text-xs">
                                    {scan.laterality === 'OD' ? 'Right Eye' : 'Left Eye'}
                                  </Badge>
                                </div>
                                <p className="text-sm text-gray-500 mt-1">
                                  {scan.date} • Dr. {scan.doctor}
                                </p>
                                <div className="flex items-center gap-3 mt-2">
                                  <Badge className={`${config.bg} ${config.color} border-0`}>
                                    {scan.result}
                                  </Badge>
                                  <span className="text-xs text-gray-500">
                                    Confidence: {scan.confidence}%
                                  </span>
                                </div>
                              </div>
                            </div>
                            <ChevronRight className={`w-5 h-5 text-gray-400 transition-transform ${
                              selectedScan === scan.id ? 'rotate-90' : ''
                            }`} />
                          </div>

                          {/* Expanded Details */}
                          {selectedScan === scan.id && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: 'auto' }}
                              className="mt-4 pt-4 border-t border-slate-100"
                            >
                              <div className="grid md:grid-cols-2 gap-4">
                                <div>
                                  <p className="text-sm font-medium text-gray-700 mb-1">Doctor's Notes</p>
                                  <p className="text-sm text-gray-600 bg-slate-50 p-3 rounded-lg">
                                    {scan.notes}
                                  </p>
                                </div>
                                <div>
                                  <p className="text-sm font-medium text-gray-700 mb-1">AI Analysis Details</p>
                                  <div className="space-y-2">
                                    <div className="flex justify-between text-sm">
                                      <span className="text-gray-600">Severity Level</span>
                                      <span className={`font-medium ${config.color}`}>{scan.severity}/4</span>
                                    </div>
                                    <Progress value={(scan.severity / 4) * 100} className="h-2" />
                                    <div className="flex justify-between text-sm">
                                      <span className="text-gray-600">AI Confidence</span>
                                      <span className="font-medium text-cyan-600">{scan.confidence}%</span>
                                    </div>
                                    <Progress value={scan.confidence} className="h-2" />
                                  </div>
                                </div>
                              </div>
                              <div className="flex gap-2 mt-4">
                                <Button size="sm" variant="outline">
                                  <Download className="w-4 h-4 mr-1" />
                                  Download PDF
                                </Button>
                                <Button size="sm" variant="outline">
                                  <Printer className="w-4 h-4 mr-1" />
                                  Print
                                </Button>
                              </div>
                            </motion.div>
                          )}
                        </CardContent>
                      </Card>
                    </motion.div>
                  );
                })}
              </div>

              {/* Summary Sidebar */}
              <div className="space-y-4">
                <Card className="bg-gradient-to-br from-cyan-500 to-blue-600 text-white border-0">
                  <CardContent className="p-5">
                    <Shield className="w-8 h-8 mb-3 opacity-80" />
                    <h3 className="font-semibold text-lg mb-1">Your Data is Secure</h3>
                    <p className="text-sm opacity-90">
                      All your medical data is encrypted and stored in compliance with HIPAA regulations.
                    </p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">Quick Actions</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <Button
                      variant="outline"
                      className="w-full justify-start"
                      onClick={() => setActiveTab('booking')}
                    >
                      <Calendar className="w-4 h-4 mr-2" />
                      Book Follow-up
                    </Button>
                    <Button
                      variant="outline"
                      className="w-full justify-start"
                      onClick={() => setActiveTab('timeline')}
                    >
                      <TrendingUp className="w-4 h-4 mr-2" />
                      View Timeline
                    </Button>
                    <Button
                      variant="outline"
                      className="w-full justify-start"
                      onClick={() => setActiveTab('education')}
                    >
                      <BookOpen className="w-4 h-4 mr-2" />
                      Learn More
                    </Button>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">Contact Info</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex items-center gap-2 text-sm">
                      <User className="w-4 h-4 text-gray-400" />
                      <span>{PATIENT_DATA.primaryDoctor}</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <Phone className="w-4 h-4 text-gray-400" />
                      <span>{PATIENT_DATA.phone}</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <Mail className="w-4 h-4 text-gray-400" />
                      <span>{PATIENT_DATA.email}</span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Book Appointment Tab */}
          <TabsContent value="booking">
            {bookingConfirmed ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="max-w-lg mx-auto text-center py-16"
              >
                <div className="w-20 h-20 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-6">
                  <CheckCircle className="w-10 h-10 text-emerald-600" />
                </div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Appointment Confirmed!</h2>
                <p className="text-gray-600 mb-6">
                  Your appointment with {DOCTORS.find(d => d.id === selectedDoctor)?.name} is scheduled for tomorrow at {selectedSlot}.
                </p>
                <p className="text-sm text-gray-500">
                  A confirmation has been sent to your email.
                </p>
              </motion.div>
            ) : (
              <div className="grid lg:grid-cols-3 gap-6">
                {/* Doctor Selection */}
                <div className="lg:col-span-2 space-y-4">
                  <h2 className="text-lg font-semibold text-gray-900 mb-4">Select a Doctor</h2>
                  {DOCTORS.map((doctor, index) => (
                    <motion.div
                      key={doctor.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                    >
                      <Card
                        className={`cursor-pointer transition-all ${
                          selectedDoctor === doctor.id
                            ? 'ring-2 ring-cyan-500 shadow-md'
                            : 'hover:shadow-sm'
                        }`}
                        onClick={() => {
                          setSelectedDoctor(doctor.id);
                          setSelectedSlot(null);
                        }}
                      >
                        <CardContent className="p-5">
                          <div className="flex items-center gap-4">
                            <div className="w-14 h-14 bg-gradient-to-br from-cyan-100 to-blue-100 rounded-full flex items-center justify-center text-lg font-bold text-cyan-700">
                              {doctor.image}
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <h3 className="font-semibold text-gray-900">{doctor.name}</h3>
                                <div className="flex items-center gap-1">
                                  <Star className="w-4 h-4 text-amber-400 fill-amber-400" />
                                  <span className="text-sm font-medium">{doctor.rating}</span>
                                  <span className="text-xs text-gray-500">({doctor.reviews} reviews)</span>
                                </div>
                              </div>
                              <p className="text-sm text-gray-500">{doctor.specialty}</p>
                            </div>
                            <ChevronRight className={`w-5 h-5 text-gray-400 transition-transform ${
                              selectedDoctor === doctor.id ? 'rotate-90' : ''
                            }`} />
                          </div>

                          {/* Time Slots */}
                          {selectedDoctor === doctor.id && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: 'auto' }}
                              className="mt-4 pt-4 border-t border-slate-100"
                            >
                              <p className="text-sm font-medium text-gray-700 mb-3">
                                Available Tomorrow (April 19)
                              </p>
                              <div className="flex flex-wrap gap-2">
                                {doctor.availableSlots.map((slot) => (
                                  <button
                                    key={slot}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setSelectedSlot(slot);
                                    }}
                                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                                      selectedSlot === slot
                                        ? 'bg-cyan-500 text-white'
                                        : 'bg-slate-100 text-gray-700 hover:bg-slate-200'
                                    }`}
                                  >
                                    {slot}
                                  </button>
                                ))}
                              </div>
                            </motion.div>
                          )}
                        </CardContent>
                      </Card>
                    </motion.div>
                  ))}

                  {/* Book Button */}
                  {selectedDoctor && selectedSlot && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="pt-4"
                    >
                      <Button
                        size="lg"
                        className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700"
                        onClick={handleBookAppointment}
                      >
                        <Calendar className="w-5 h-5 mr-2" />
                        Confirm Appointment
                      </Button>
                    </motion.div>
                  )}
                </div>

                {/* Booking Info */}
                <div className="space-y-4">
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm">Booking Information</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3 text-sm">
                      <div className="flex items-start gap-2">
                        <MapPin className="w-4 h-4 text-gray-400 mt-0.5" />
                        <div>
                          <p className="font-medium">Boston Eye Institute</p>
                          <p className="text-gray-500">123 Medical Center Dr, Boston, MA</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Clock className="w-4 h-4 text-gray-400" />
                        <span className="text-gray-600">Duration: 30 minutes</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Phone className="w-4 h-4 text-gray-400" />
                        <span className="text-gray-600">(555) 123-4567</span>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-amber-50 border-amber-200">
                    <CardContent className="p-4">
                      <div className="flex items-start gap-2">
                        <AlertCircle className="w-5 h-5 text-amber-600 mt-0.5" />
                        <div>
                          <p className="font-medium text-amber-800 text-sm">Before Your Visit</p>
                          <ul className="text-xs text-amber-700 mt-1 space-y-1">
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
            )}
          </TabsContent>

          {/* Health Timeline Tab */}
          <TabsContent value="timeline">
            <div className="grid lg:grid-cols-3 gap-6">
              {/* Chart */}
              <div className="lg:col-span-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="w-5 h-5 text-cyan-600" />
                      Disease Severity Over Time
                    </CardTitle>
                    <CardDescription>
                      Tracking your retinal health across all screenings
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={TREND_DATA}>
                          <defs>
                            <linearGradient id="colorSeverity" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#06B6D4" stopOpacity={0.3} />
                              <stop offset="95%" stopColor="#06B6D4" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                          <XAxis dataKey="date" stroke="#94A3B8" fontSize={12} />
                          <YAxis stroke="#94A3B8" fontSize={12} domain={[0, 4]} ticks={[0, 1, 2, 3, 4]} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: '#fff',
                              border: '1px solid #E2E8F0',
                              borderRadius: '8px',
                              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                            }}
                          />
                          <Area
                            type="monotone"
                            dataKey="severity"
                            stroke="#06B6D4"
                            fillOpacity={1}
                            fill="url(#colorSeverity)"
                            strokeWidth={2}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Severity Legend */}
                    <div className="flex flex-wrap gap-4 mt-4 justify-center">
                      {Object.entries(SEVERITY_CONFIG).map(([level, config]) => (
                        <div key={level} className="flex items-center gap-2">
                          <div className={`w-3 h-3 rounded-full ${config.bg.replace('bg-', 'bg-').replace('50', '400')}`} />
                          <span className="text-xs text-gray-600">{config.label} (Grade {level})</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Timeline Events */}
              <div className="space-y-4">
                <h3 className="font-semibold text-gray-900">Screening History</h3>
                {SCAN_HISTORY.map((scan, index) => {
                  const config = SEVERITY_CONFIG[scan.severity] || SEVERITY_CONFIG[0];
                  return (
                    <motion.div
                      key={scan.id}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="flex gap-3"
                    >
                      <div className="flex flex-col items-center">
                        <div className={`w-3 h-3 rounded-full ${config.bg.replace('bg-', 'bg-').replace('50', '400')}`} />
                        {index < SCAN_HISTORY.length - 1 && (
                          <div className="w-0.5 h-12 bg-slate-200" />
                        )}
                      </div>
                      <div className="pb-4">
                        <p className="text-sm font-medium text-gray-900">{scan.date}</p>
                        <p className={`text-sm ${config.color}`}>{scan.result}</p>
                        <p className="text-xs text-gray-500">{scan.doctor}</p>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          </TabsContent>

          {/* Education Tab */}
          <TabsContent value="education">
            <div className="mb-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-2">Educational Resources</h2>
              <p className="text-gray-600">Learn more about your condition and how to protect your vision</p>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              {EDUCATIONAL_CONTENT.map((content, index) => (
                <motion.div
                  key={content.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Card className="overflow-hidden hover:shadow-lg transition-shadow cursor-pointer group">
                    {/* Thumbnail */}
                    <div className="h-40 bg-gradient-to-br from-cyan-100 to-blue-100 flex items-center justify-center relative overflow-hidden">
                      {content.type === 'video' ? (
                        <>
                          <Video className="w-12 h-12 text-cyan-400" />
                          <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/20">
                            <div className="w-14 h-14 bg-white/90 rounded-full flex items-center justify-center">
                              <Play className="w-6 h-6 text-cyan-600 ml-1" />
                            </div>
                          </div>
                          <div className="absolute bottom-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                            {content.duration}
                          </div>
                        </>
                      ) : (
                        <BookOpen className="w-12 h-12 text-cyan-400" />
                      )}
                      <Badge className="absolute top-2 left-2 bg-white/90 text-gray-700">
                        {content.category}
                      </Badge>
                    </div>

                    <CardContent className="p-5">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="outline" className="text-xs">
                          {content.type === 'video' ? (
                            <Video className="w-3 h-3 mr-1" />
                          ) : (
                            <BookOpen className="w-3 h-3 mr-1" />
                          )}
                          {content.type}
                        </Badge>
                        <span className="text-xs text-gray-500">
                          {content.readTime || content.duration}
                        </span>
                      </div>
                      <h3 className="font-semibold text-gray-900 mb-2 group-hover:text-cyan-600 transition-colors">
                        {content.title}
                      </h3>
                      <p className="text-sm text-gray-600">{content.description}</p>
                      <div className="flex items-center gap-1 mt-3 text-cyan-600 text-sm font-medium">
                        <span>Read more</span>
                        <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>

            {/* Personalized Tip */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="mt-8"
            >
              <Card className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white border-0">
                <CardContent className="p-6">
                  <div className="flex items-start gap-4">
                    <Sparkles className="w-8 h-8 flex-shrink-0" />
                    <div>
                      <h3 className="font-semibold text-lg mb-2">Personalized for You</h3>
                      <p className="text-sm opacity-90 leading-relaxed">
                        Based on your recent screening showing Mild DR, we recommend reviewing our article
                        on "Managing Diabetic Retinopathy" and scheduling a follow-up within 6 months.
                        Maintaining stable blood glucose levels is key to preventing progression.
                      </p>
                      <Button
                        variant="secondary"
                        size="sm"
                        className="mt-4 bg-white/20 hover:bg-white/30 text-white border-0"
                      >
                        View Personalized Resources
                        <ArrowRight className="w-4 h-4 ml-1" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
