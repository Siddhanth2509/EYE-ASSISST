import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Users } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { toast } from 'sonner';

const ROLES = ['patient', 'doctor', 'technician', 'admin'];
const roleColors: Record<string, string> = {
  patient: 'text-emerald-400 bg-emerald-500/10',
  doctor: 'text-blue-400 bg-blue-500/10',
  technician: 'text-amber-400 bg-amber-500/10',
  admin: 'text-purple-400 bg-purple-500/10',
};

const DEMO_USERS = [
  { id: 'USR-001', name: 'Dr. Sarah Mitchell', email: 'sarah@eyeassist.ai', role: 'doctor',     phone: '+91 98001 11001', active: true, createdAt: '2026-01-15' },
  { id: 'USR-002', name: 'Ravi Kumar',         email: 'ravi@eyeassist.ai',  role: 'technician', phone: '+91 98001 11002', active: true, createdAt: '2026-02-01' },
  { id: 'USR-003', name: 'Priya Sharma',       email: 'priya@demo.com',     role: 'patient',    phone: '+91 98001 11003', active: true, createdAt: '2026-03-10' },
  { id: 'USR-004', name: 'Admin User',         email: 'admin@eyeassist.ai', role: 'admin',      phone: '+91 98001 11004', active: true, createdAt: '2026-01-01' },
];

function loadUsers(): any[] {
  try {
    const stored = JSON.parse(localStorage.getItem('eye_users') || 'null');
    if (stored && stored.length) return stored;
  } catch { /* ignore */ }
  localStorage.setItem('eye_users', JSON.stringify(DEMO_USERS));
  return DEMO_USERS;
}

export default function UserManagementPanel() {
  const [users, setUsers] = useState<any[]>(loadUsers);
  const [filterRole, setFilterRole] = useState('all');
  const [search, setSearch] = useState('');
  const [showAdd, setShowAdd] = useState(false);
  const [newUser, setNewUser] = useState({ name: '', email: '', role: 'doctor', phone: '', password: '' });

  const save = (updated: any[]) => {
    setUsers(updated);
    localStorage.setItem('eye_users', JSON.stringify(updated));
  };

  const removeUser = (id: string) => {
    if (!window.confirm('Remove this user?')) return;
    save(users.filter(u => u.id !== id));
    toast.success('User removed');
  };

  const toggleActive = (id: string) => {
    save(users.map(u => u.id === id ? { ...u, active: !u.active } : u));
    toast.success('Status updated');
  };

  const resetPwd = (u: any) => toast.success(`Password reset link sent to ${u.email}`);

  const addUser = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!newUser.name || !newUser.email) { toast.error('Name and email are required'); return; }
    const entry = {
      ...newUser,
      password: newUser.password || 'TempPass123!',
      email: newUser.email.trim(),
      id: `USR-${String(Date.now()).slice(-4)}`,
      active: true,
      createdAt: new Date().toISOString().split('T')[0],
    };
    save([...users, entry]);
    setNewUser({ name: '', email: '', role: 'doctor', phone: '', password: '' });
    setShowAdd(false);
    toast.success('User created');
  };

  const filtered = users.filter(u =>
    (filterRole === 'all' || u.role === filterRole) &&
    (u.name.toLowerCase().includes(search.toLowerCase()) ||
      u.email.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="h-[calc(100vh-56px)] overflow-auto p-4 sm:p-6">
      <div className="max-w-5xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">User Management</h2>
            <p className="text-muted-foreground text-sm">{users.length} registered users</p>
          </div>
          <Button onClick={() => setShowAdd(!showAdd)} className="btn-medical">
            <Users className="w-4 h-4 mr-2" />Add User
          </Button>
        </div>

        {/* Add User Form */}
        <AnimatePresence>
          {showAdd && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden"
            >
              <Card className="medical-panel border-primary/30">
                <CardHeader><CardTitle className="text-sm">New User</CardTitle></CardHeader>
                <CardContent>
                  <form onSubmit={addUser}>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <div>
                      <Label className="text-xs">Full Name</Label>
                      <Input className="mt-1" value={newUser.name}
                        onChange={e => setNewUser(p => ({ ...p, name: e.target.value }))}
                        placeholder="Dr. Name" />
                    </div>
                    <div>
                      <Label className="text-xs">Email</Label>
                      <Input className="mt-1" type="email" value={newUser.email}
                        onChange={e => setNewUser(p => ({ ...p, email: e.target.value }))}
                        placeholder="user@eyeassist.ai" />
                    </div>
                    <div>
                      <Label className="text-xs">Phone</Label>
                      <Input className="mt-1" value={newUser.phone}
                        onChange={e => setNewUser(p => ({ ...p, phone: e.target.value }))}
                        placeholder="+91 98765 43210" />
                    </div>
                    <div>
                      <Label className="text-xs">Role</Label>
                      <select
                        className="mt-1 w-full px-3 py-2 text-sm bg-muted rounded-lg border border-border text-foreground"
                        value={newUser.role}
                        onChange={e => setNewUser(p => ({ ...p, role: e.target.value }))}
                      >
                        {ROLES.map(r => (
                          <option key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <Label className="text-xs">Temporary Password</Label>
                      <Input className="mt-1" type="password" value={newUser.password}
                        onChange={e => setNewUser(p => ({ ...p, password: e.target.value }))}
                        placeholder="••••••••" />
                    </div>
                    </div>
                    <div className="flex gap-2 mt-4">
                      <Button size="sm" type="submit">Create User</Button>
                      <Button size="sm" type="button" variant="outline" onClick={() => setShowAdd(false)}>Cancel</Button>
                    </div>
                  </form>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Filters */}
        <div className="flex gap-3 flex-wrap items-center">
          <Input
            className="max-w-xs h-9"
            placeholder="Search name or email..."
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
          <div className="flex gap-1 flex-wrap">
            {['all', ...ROLES].map(r => (
              <button key={r} onClick={() => setFilterRole(r)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all capitalize ${
                  filterRole === r
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-muted-foreground hover:text-foreground'
                }`}>
                {r}
              </button>
            ))}
          </div>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {ROLES.map(r => {
            const count = users.filter(u => u.role === r).length;
            return (
              <div key={r} className={`p-3 rounded-xl border border-border bg-card flex items-center gap-3`}>
                <div className={`text-xs px-2 py-1 rounded-full capitalize font-semibold ${roleColors[r]}`}>{r}</div>
                <span className="text-xl font-bold text-foreground">{count}</span>
              </div>
            );
          })}
        </div>

        {/* User Table */}
        <Card className="medical-panel overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-4 text-xs text-muted-foreground font-medium uppercase tracking-wider">User</th>
                  <th className="text-left p-4 text-xs text-muted-foreground font-medium uppercase tracking-wider">Role</th>
                  <th className="text-left p-4 text-xs text-muted-foreground font-medium uppercase tracking-wider hidden md:table-cell">Phone</th>
                  <th className="text-left p-4 text-xs text-muted-foreground font-medium uppercase tracking-wider hidden lg:table-cell">Joined</th>
                  <th className="text-left p-4 text-xs text-muted-foreground font-medium uppercase tracking-wider">Status</th>
                  <th className="text-right p-4 text-xs text-muted-foreground font-medium uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((u, i) => (
                  <motion.tr
                    key={u.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.04 }}
                    className="border-b border-border/50 hover:bg-muted/30 transition-colors"
                  >
                    <td className="p-4">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-xs font-bold text-primary flex-shrink-0">
                          {u.name.split(' ').map((n: string) => n[0]).join('').slice(0, 2).toUpperCase()}
                        </div>
                        <div>
                          <p className="font-medium text-foreground">{u.name}</p>
                          <p className="text-xs text-muted-foreground">{u.email}</p>
                        </div>
                      </div>
                    </td>
                    <td className="p-4">
                      <span className={`text-xs px-2 py-1 rounded-full capitalize font-medium ${roleColors[u.role] || ''}`}>
                        {u.role}
                      </span>
                    </td>
                    <td className="p-4 hidden md:table-cell text-xs text-muted-foreground">{u.phone || '—'}</td>
                    <td className="p-4 hidden lg:table-cell text-xs text-muted-foreground">{u.createdAt}</td>
                    <td className="p-4">
                      <button
                        onClick={() => toggleActive(u.id)}
                        className={`text-xs px-2 py-1 rounded-full ${u.active ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}
                      >
                        {u.active ? 'Active' : 'Inactive'}
                      </button>
                    </td>
                    <td className="p-4">
                      <div className="flex items-center justify-end gap-2">
                        <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => resetPwd(u)}>
                          Reset Pwd
                        </Button>
                        <Button
                          size="sm" variant="outline"
                          className="h-7 text-xs text-red-400 border-red-400/30 hover:bg-red-500/10"
                          onClick={() => removeUser(u.id)}
                        >
                          Remove
                        </Button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={6} className="p-8 text-center text-muted-foreground text-sm">
                      No users match your search
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}
