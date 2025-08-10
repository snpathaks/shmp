function mobileForm() {
    return {
        db: null,
        isOnline: true,
        pendingCount: 0,
        syncing: false,
        message: '',
        error: false,
        formData: {
            soldier_id: 'f3e9b8c0-2c7a-4b6d-8a1e-5b9c1f0d7e6a',
            sleep_hours: null,
            stress_level: null,
        },

        init() {
            this.db = new Dexie('SHMPDatabase');
            this.db.version(1).stores({
                mood_reports: '++id, soldier_id, date, stress_level, sleep_hours'
            });

            this.updateOnlineStatus();
            window.addEventListener('online', () => this.updateOnlineStatus());
            window.addEventListener('offline', () => this.updateOnlineStatus());
            this.updatePendingCount();
        },

        updateOnlineStatus() {
            this.isOnline = navigator.onLine;
            console.log(`Status changed: ${this.isOnline ? 'Online' : 'Offline'}`);
        },

        async updatePendingCount() {
            this.pendingCount = await this.db.mood_reports.count();
        },

        async saveReport() {
            if (!this.formData.stress_level) {
                this.showMessage('Please select a stress level.', true);
                return;
            }

            const report = {
                soldier_id: this.formData.soldier_id,
                date: new Date().toISOString(),
                sleep_hours: parseFloat(this.formData.sleep_hours),
                stress_level: parseInt(this.formData.stress_level),
            };

            try {
                await this.db.mood_reports.add(report);
                this.showMessage('Report saved locally!', false);
                this.resetForm();
                this.updatePendingCount();
            } catch (e) {
                console.error("Failed to save report to IndexedDB", e);
                this.showMessage('Error saving report.', true);
            }
        },

        async syncData() {
            if (!this.isOnline || this.pendingCount === 0 || this.syncing) return;

            this.syncing = true;
            this.message = '';

            try {
                const reportsToSync = await this.db.mood_reports.toArray();
                const token = this.getAuthToken(); // You need a way to get the auth token

                if (!token) {
                    this.showMessage('Authentication error. Please log in again on the main site.', true);
                    this.syncing = false;
                    return;
                }

                const response = await fetch('/api/sync', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': token
                    },
                    body: JSON.stringify({ mood_reports: reportsToSync })
                });

                if (response.ok) {
                    // Clear the local database on successful sync
                    await this.db.mood_reports.clear();
                    this.showMessage(`Successfully synced ${reportsToSync.length} reports.`, false);
                    this.updatePendingCount();
                } else {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Sync failed on server.');
                }

            } catch (e) {
                console.error("Sync failed", e);
                this.showMessage(`Sync failed: ${e.message}`, true);
            } finally {
                this.syncing = false;
            }
        },

        resetForm() {
            this.formData.sleep_hours = null;
            this.formData.stress_level = null;
        },

        showMessage(msg, isError = false) {
            this.message = msg;
            this.error = isError;
            setTimeout(() => { this.message = ''; }, 4000);
        },

        getAuthToken() {
            // This is a simplified way to get the token.
            // In a real app, the token might be stored more securely or refreshed.
            // We assume the cookie is accessible for this demo.
            const cookies = document.cookie.split('; ').reduce((acc, cookie) => {
                const [key, value] = cookie.split('=');
                acc[key] = value;
                return acc;
            }, {});
            return cookies.access_token ? decodeURIComponent(cookies.access_token) : null;
        }
    };
}