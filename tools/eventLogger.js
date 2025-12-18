const mqtt = require('mqtt');
const fs = require('fs');
const path = require('path');

// Load configuration from file
const configPath = path.join(__dirname, 'config.json');
let config;

try {
    const configFile = fs.readFileSync(configPath, 'utf8');
    config = JSON.parse(configFile);
    console.log('Configuration loaded from config.json');
} catch (err) {
    console.error('Error loading config.json, using defaults:', err.message);
    // Default configuration if file doesn't exist
    config = {
        mqttBroker: 'mqtt://localhost:1883',
        frigateEventsTopic: 'frigate/events',
        logFile: 'frigate-events.log',
        cameras: []
    };

    // Create default config file
    try {
        fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
        console.log('Created default config.json file');
    } catch (writeErr) {
        console.error('Could not create config.json:', writeErr.message);
    }
}

// Create log file path
const logPath = path.join(__dirname, config.logFile);

// Function to write to log file
function logEvent(message) {
    const timestamp = new Date().toISOString();
    const logEntry = `[${timestamp}] ${message}\n`;

    fs.appendFile(logPath, logEntry, (err) => {
        if (err) {
            console.error('Error writing to log file:', err);
        }
    });

    // Also log to console
    console.log(logEntry.trim());
}

// MQTT connection options
const mqttOptions = {
    clientId: `frigate-logger-${Math.random().toString(16).slice(2, 8)}`,
}

// Connect to MQTT broker
console.log(`Connecting to MQTT broker: ${config.mqttBroker}`);
const client = mqtt.connect(config.mqttBroker, mqttOptions);

// Handle connection
client.on('connect', () => {
    console.log('Connected to MQTT broker');
    logEvent('=== MQTT Logger Started ===');
    if (config.cameras.length > 0) {
        logEvent(`Filtering for cameras: ${config.cameras.join(', ')}`);
    } else {
        logEvent('Logging all cameras');
    }

    // Subscribe to Frigate events topic
    client.subscribe(config.frigateEventsTopic, (err) => {
        if (err) {
            console.error('Subscription error:', err);
            logEvent(`ERROR: Failed to subscribe to ${config.frigateEventsTopic}`);
        } else {
            console.log(`Subscribed to: ${config.frigateEventsTopic}`);
            logEvent(`Subscribed to: ${config.frigateEventsTopic}`);
        }
    });
});

// Handle incoming messages
client.on('message', (topic, message) => {
    try {
        const payload = message.toString();
        let formattedMessage;
        let shouldLog = true;

        try {
            // Try to parse as JSON for better formatting
            const jsonData = JSON.parse(payload);

            // Filter by camera if cameras are specified
            if (config.cameras.length > 0) {
                const camera = jsonData.after?.camera || jsonData.before?.camera;
                if (!camera || !config.cameras.includes(camera)) {
                    shouldLog = false;
                }
            }

            if (shouldLog) {
                formattedMessage = `Topic: ${topic}\nData: ${JSON.stringify(jsonData, null, 2)}`;
            }
        } catch (e) {
            // If not JSON, log as plain text
            formattedMessage = `Topic: ${topic}\nData: ${payload}`;
        }

        if (shouldLog && formattedMessage) {
            logEvent(formattedMessage);
            logEvent('---');
        }
    } catch (err) {
        console.error('Error processing message:', err);
    }
});

// Handle errors
client.on('error', (err) => {
    console.error('MQTT Error:', err);
    logEvent(`ERROR: ${err.message}`);
});

// Handle disconnection
client.on('close', () => {
    console.log('Disconnected from MQTT broker');
    logEvent('=== MQTT Logger Stopped ===');
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\nShutting down...');
    client.end();
    process.exit(0);
});

console.log(`Logging events to: ${logPath}`);
console.log('Press Ctrl+C to stop\n');