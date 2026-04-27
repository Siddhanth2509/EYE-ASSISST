import { useState, useEffect } from 'react';
import { Shield, Key, CheckCircle, Save, Eye, EyeOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { toast } from 'sonner';

export default function AdminSettingsPanel() {
  const [adminKey, setAdminKey] = useState('');
  const [isSaved, setIsSaved] = useState(false);
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    const key = localStorage.getItem('eye_admin_key') || 'EYEADMIN2026';
    setAdminKey(key);
  }, []);

  const handleSave = () => {
    if (adminKey.length < 8) {
      toast.error('Admin key must be at least 8 characters long');
      return;
    }
    localStorage.setItem('eye_admin_key', adminKey);
    setIsSaved(true);
    toast.success('Admin key updated successfully');
    setTimeout(() => setIsSaved(false), 3000);
  };

  return (
    <div className="h-[calc(100vh-56px)] overflow-auto p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Shield className="w-6 h-6 text-purple-500" />
            System Settings
          </h2>
          <p className="text-muted-foreground mt-1">Configure global platform security and administrative parameters</p>
        </div>

        <Card className="medical-panel border-purple-500/20">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Key className="w-5 h-5 text-primary" />
              Administrative Access Key
            </CardTitle>
            <CardDescription>
              This secret key allows instant administrator login via the Ctrl+Shift+A shortcut on the login page.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4 max-w-md">
              <div className="space-y-2">
                <Label>Secret Key</Label>
                <div className="flex gap-3">
                  <div className="relative flex-1">
                    <Input 
                      type={showKey ? "text" : "password"} 
                      value={adminKey}
                      onChange={(e) => setAdminKey(e.target.value)}
                      placeholder="Enter new admin key"
                      className="pr-10"
                    />
                    <button
                      type="button"
                      onClick={() => setShowKey(!showKey)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                    >
                      {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                  <Button onClick={handleSave} className="bg-purple-600 hover:bg-purple-700 min-w-[120px]">
                    {isSaved ? <><CheckCircle className="w-4 h-4 mr-2" /> Saved</> : <><Save className="w-4 h-4 mr-2" /> Save Key</>}
                  </Button>
                </div>
              </div>
              <p className="text-xs text-muted-foreground">
                Keep this key secure. Anyone with this key can access the admin dashboard without a registered account.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
