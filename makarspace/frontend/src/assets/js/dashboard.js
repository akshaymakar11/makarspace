// MakarSpace Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Space Background with Particles.js
    initParticles();
    
    // Initialize Charts
    initCharts();
    
    // Setup Navigation
    setupNavigation();
    
    // Initialize 3D Spacecraft Model (if Three.js is available)
    if (typeof THREE !== 'undefined') {
        initSpacecraftModel();
    }
    
    // Show random anomaly detection animation
    setInterval(showRandomAnomaly, 30000); // Every 30 seconds
});

// Initialize Particles.js Background
function initParticles() {
    particlesJS('particles-js', {
        "particles": {
            "number": {
                "value": 100,
                "density": {
                    "enable": true,
                    "value_area": 800
                }
            },
            "color": {
                "value": ["#ffffff", "#1E5CB3", "#FC3D21"]
            },
            "shape": {
                "type": "circle"
            },
            "opacity": {
                "value": 0.5,
                "random": true,
                "anim": {
                    "enable": true,
                    "speed": 1,
                    "opacity_min": 0.1,
                    "sync": false
                }
            },
            "size": {
                "value": 2,
                "random": true,
                "anim": {
                    "enable": true,
                    "speed": 2,
                    "size_min": 0.1,
                    "sync": false
                }
            },
            "line_linked": {
                "enable": true,
                "distance": 150,
                "color": "#1E5CB3",
                "opacity": 0.2,
                "width": 1
            },
            "move": {
                "enable": true,
                "speed": 0.3,
                "direction": "none",
                "random": true,
                "straight": false,
                "out_mode": "out",
                "bounce": false
            }
        },
        "interactivity": {
            "detect_on": "canvas",
            "events": {
                "onhover": {
                    "enable": true,
                    "mode": "grab"
                },
                "onclick": {
                    "enable": true,
                    "mode": "push"
                },
                "resize": true
            },
            "modes": {
                "grab": {
                    "distance": 140,
                    "line_linked": {
                        "opacity": 0.4
                    }
                },
                "push": {
                    "particles_nb": 3
                }
            }
        },
        "retina_detect": true
    });
}

// Initialize all Chart.js charts
function initCharts() {
    // Common chart options
    const commonOptions = {
        maintainAspectRatio: false,
        responsive: true,
        animation: {
            duration: 2000,
            easing: 'easeOutQuart'
        },
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                backgroundColor: 'rgba(18, 18, 35, 0.8)',
                titleColor: '#ffffff',
                bodyColor: '#e0e0e0',
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 1,
                padding: 10,
                boxPadding: 5,
                usePointStyle: true,
                titleFont: {
                    family: "'Exo 2', sans-serif",
                    size: 14
                },
                bodyFont: {
                    family: "'Roboto', sans-serif",
                    size: 12
                }
            }
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                    borderColor: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: 'rgba(224, 224, 224, 0.8)',
                    font: {
                        family: "'Roboto', sans-serif",
                        size: 10
                    }
                }
            },
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                    borderColor: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: 'rgba(224, 224, 224, 0.8)',
                    font: {
                        family: "'Roboto', sans-serif",
                        size: 10
                    }
                }
            }
        }
    };

    // Anomaly Detection Chart
    if (document.getElementById('anomalyChart')) {
        const anomalyCtx = document.getElementById('anomalyChart').getContext('2d');
        const anomalyChart = new Chart(anomalyCtx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'Detection Accuracy',
                    data: [97.2, 98.1, 98.5, 99.0, 99.2, 99.4],
                    borderColor: '#1E5CB3',
                    backgroundColor: 'rgba(30, 92, 179, 0.2)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: { ...commonOptions }
        });
    }

    // Maintenance Chart
    if (document.getElementById('maintenanceChart')) {
        const maintenanceCtx = document.getElementById('maintenanceChart').getContext('2d');
        const maintenanceChart = new Chart(maintenanceCtx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'Prediction Accuracy',
                    data: [92.5, 94.1, 95.5, 96.2, 97.0, 97.8],
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.2)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: { ...commonOptions }
        });
    }

    // Latency Chart
    if (document.getElementById('latencyChart')) {
        const latencyCtx = document.getElementById('latencyChart').getContext('2d');
        const latencyChart = new Chart(latencyCtx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'Response Time (ms)',
                    data: [45, 40, 35, 30, 26, 23],
                    borderColor: '#FF9800',
                    backgroundColor: 'rgba(255, 152, 0, 0.2)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: { ...commonOptions }
        });
    }

    // Explainability Chart
    if (document.getElementById('explainabilityChart')) {
        const explainabilityCtx = document.getElementById('explainabilityChart').getContext('2d');
        const explainabilityChart = new Chart(explainabilityCtx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'Clarity Score',
                    data: [85.5, 88.2, 90.0, 92.1, 93.5, 94.1],
                    borderColor: '#9C27B0',
                    backgroundColor: 'rgba(156, 39, 176, 0.2)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: { ...commonOptions }
        });
    }

    // System Health Charts
    const systemChartIds = ['powerSystemChart', 'propulsionChart', 'commsChart', 'thermalChart'];
    
    systemChartIds.forEach(chartId => {
        if (document.getElementById(chartId)) {
            const ctx = document.getElementById(chartId).getContext('2d');
            const data = generateRandomChartData();
            
            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
                    datasets: [{
                        label: 'System Health',
                        data: data,
                        borderColor: chartId === 'propulsionChart' ? '#FF9800' : '#1E5CB3',
                        backgroundColor: chartId === 'propulsionChart' ? 'rgba(255, 152, 0, 0.2)' : 'rgba(30, 92, 179, 0.2)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: { ...commonOptions }
            });
        }
    });
}

// Generate random chart data with a trend
function generateRandomChartData() {
    const baseValue = 70 + Math.random() * 20;
    const variance = 5;
    const trend = Math.random() > 0.5 ? 1 : -1;
    
    return Array(6).fill(0).map((_, i) => {
        return baseValue + trend * (i * 0.5) + (Math.random() * variance * 2 - variance);
    });
}

// Setup navigation between dashboard pages
function setupNavigation() {
    const navLinks = document.querySelectorAll('.sidebar-link');
    const pages = document.querySelectorAll('.dashboard-page');
    const currentSection = document.getElementById('current-section');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all links
            navLinks.forEach(l => l.classList.remove('active'));
            
            // Add active class to clicked link
            this.classList.add('active');
            
            // Hide all pages
            pages.forEach(page => page.classList.remove('active'));
            
            // Show the selected page
            const pageId = this.getAttribute('data-page');
            const selectedPage = document.getElementById(pageId);
            
            if (selectedPage) {
                selectedPage.classList.add('active');
                currentSection.textContent = this.querySelector('span').textContent;
            }
        });
    });
}

// Initialize 3D Spacecraft Model with Three.js
function initSpacecraftModel() {
    const container = document.getElementById('spacecraft-visualization');
    
    if (!container) return;
    
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);
    
    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // Create a simple spacecraft model
    const spacecraftGeometry = new THREE.Group();
    
    // Main body
    const bodyGeometry = new THREE.CylinderGeometry(1, 1, 4, 16);
    const bodyMaterial = new THREE.MeshPhongMaterial({ color: 0x333333 });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    spacecraftGeometry.add(body);
    
    // Solar panels
    const panelGeometry = new THREE.BoxGeometry(6, 0.1, 1.5);
    const panelMaterial = new THREE.MeshPhongMaterial({ color: 0x1E5CB3 });
    const panel1 = new THREE.Mesh(panelGeometry, panelMaterial);
    panel1.position.x = 3;
    spacecraftGeometry.add(panel1);
    
    const panel2 = new THREE.Mesh(panelGeometry, panelMaterial);
    panel2.position.x = -3;
    spacecraftGeometry.add(panel2);
    
    // Antenna
    const antennaGeometry = new THREE.CylinderGeometry(0.05, 0.05, 2, 8);
    const antennaMaterial = new THREE.MeshPhongMaterial({ color: 0xcccccc });
    const antenna = new THREE.Mesh(antennaGeometry, antennaMaterial);
    antenna.position.y = 3;
    spacecraftGeometry.add(antenna);
    
    // Add spacecraft to scene
    scene.add(spacecraftGeometry);
    
    // Position camera
    camera.position.z = 10;
    
    // Animation function
    function animate() {
        requestAnimationFrame(animate);
        
        // Rotate spacecraft slowly
        spacecraftGeometry.rotation.y += 0.005;
        spacecraftGeometry.rotation.x = Math.sin(Date.now() * 0.001) * 0.1;
        
        renderer.render(scene, camera);
    }
    
    // Start animation
    animate();
    
    // Resize handler
    window.addEventListener('resize', function() {
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        renderer.setSize(width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
    });
}

// Show random anomaly detection animation
function showRandomAnomaly() {
    // Only show if on the overview page
    if (!document.getElementById('overview').classList.contains('active')) return;
    
    // Create anomaly detection indicator
    const anomalyIndicator = document.createElement('div');
    anomalyIndicator.className = 'anomaly-indicator';
    anomalyIndicator.innerHTML = `
        <div class="anomaly-pulse"></div>
        <div class="anomaly-content">
            <div class="anomaly-title">Anomaly Detected</div>
            <div class="anomaly-details">Minor radiation fluctuation in sector 7G</div>
        </div>
    `;
    
    // Add to document
    document.body.appendChild(anomalyIndicator);
    
    // Show animation
    setTimeout(() => {
        anomalyIndicator.classList.add('show');
        
        // Play alert sound
        const alertSound = new Audio('../assets/sounds/alert.mp3');
        alertSound.volume = 0.3;
        alertSound.play().catch(e => {
            // Ignore errors - browser might block autoplay
            console.log('Audio autoplay was prevented');
        });
        
        // Remove after delay
        setTimeout(() => {
            anomalyIndicator.classList.remove('show');
            setTimeout(() => {
                anomalyIndicator.remove();
            }, 1000);
        }, 5000);
    }, 100);
}

// Add dynamic CSS for anomaly indicator
function addAnomalyStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .anomaly-indicator {
            position: fixed;
            bottom: -100px;
            right: 20px;
            background: rgba(18, 18, 35, 0.9);
            border: 1px solid rgba(252, 61, 33, 0.5);
            border-radius: 8px;
            padding: 15px;
            display: flex;
            align-items: center;
            gap: 15px;
            box-shadow: 0 0 15px rgba(252, 61, 33, 0.3);
            z-index: 1000;
            transition: bottom 0.5s ease;
            backdrop-filter: blur(10px);
        }
        
        .anomaly-indicator.show {
            bottom: 20px;
        }
        
        .anomaly-pulse {
            width: 24px;
            height: 24px;
            background-color: var(--accent);
            border-radius: 50%;
            position: relative;
        }
        
        .anomaly-pulse::before {
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background-color: var(--accent);
            animation: pulse 2s infinite;
            box-shadow: 0 0 10px var(--accent);
        }
        
        .anomaly-content {
            display: flex;
            flex-direction: column;
        }
        
        .anomaly-title {
            font-weight: 600;
            color: var(--accent);
            margin-bottom: 5px;
        }
        
        .anomaly-details {
            font-size: 0.9rem;
            color: var(--space-silver);
        }
    `;
    
    document.head.appendChild(style);
}

// Add anomaly styles on load
addAnomalyStyles();

// Add a global function to simulate new anomalies (for demo purposes)
window.simulateAnomaly = function(type = 'random') {
    const anomalyTypes = [
        { title: 'Radiation Spike', details: 'Increased radiation levels detected in sector 7G' },
        { title: 'Temperature Anomaly', details: 'Thruster assembly temperature exceeding normal range' },
        { title: 'Power Fluctuation', details: 'Voltage instability in primary power distribution' },
        { title: 'Signal Degradation', details: 'Communication signal quality declining beyond threshold' },
        { title: 'Gyroscopic Drift', details: 'Minor orientation discrepancy detected in navigation' }
    ];
    
    let anomaly;
    if (type === 'random') {
        const index = Math.floor(Math.random() * anomalyTypes.length);
        anomaly = anomalyTypes[index];
    } else if (type === 'critical') {
        anomaly = { 
            title: 'CRITICAL ANOMALY', 
            details: 'Primary propulsion system showing signs of imminent failure'
        };
    }
    
    // Create anomaly detection indicator
    const anomalyIndicator = document.createElement('div');
    anomalyIndicator.className = 'anomaly-indicator';
    anomalyIndicator.innerHTML = `
        <div class="anomaly-pulse"></div>
        <div class="anomaly-content">
            <div class="anomaly-title">${anomaly.title}</div>
            <div class="anomaly-details">${anomaly.details}</div>
        </div>
    `;
    
    // Add to document
    document.body.appendChild(anomalyIndicator);
    
    // Show animation
    setTimeout(() => {
        anomalyIndicator.classList.add('show');
        
        // Remove after delay
        setTimeout(() => {
            anomalyIndicator.classList.remove('show');
            setTimeout(() => {
                anomalyIndicator.remove();
            }, 1000);
        }, 5000);
    }, 100);
    
    // If critical, update status
    if (type === 'critical') {
        const statusIcon = document.querySelector('.status-icon');
        const statusText = statusIcon.nextElementSibling;
        
        if (statusIcon && statusText) {
            statusIcon.className = 'status-icon status-critical';
            statusText.textContent = 'Critical';
            
            const statusBadge = document.querySelector('.status-badge');
            if (statusBadge) {
                statusBadge.style.background = 'rgba(244, 67, 54, 0.1)';
                statusBadge.style.borderColor = 'rgba(244, 67, 54, 0.3)';
                statusBadge.style.color = '#f44336';
            }
        }
    }
};
