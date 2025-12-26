const mqtt = require('mqtt');
const fs = require('fs');

// Read configuration
const config = JSON.parse(fs.readFileSync('config.json', 'utf8'));

// Main function
async function main() {
    // Parse command line arguments
    const camera = process.argv[2] || config.cameras[0];

    console.log('=== Motion MQTT Injector ===');
    console.log(`Camera: ${camera}`);
    console.log(`Topic: frigate/${camera}/motion`);
    console.log(`\nConnecting to MQTT broker: ${config.mqttBroker}`);

    const client = mqtt.connect(config.mqttBroker);

    client.on('connect', async () => {
        console.log('✓ Connected to MQTT broker\n');

        const topic = `frigate/${camera}/motion`;

        // Send motion ON
        console.log('[1/2] Sending motion ON...');
        client.publish(topic, 'ON');
        const onTimestamp = new Date().toISOString();
        console.log(`✓ Published motion ON at ${onTimestamp}`);

        // Wait 3 seconds
        console.log('\n⏳ Waiting 3 seconds...\n');
        await new Promise(resolve => setTimeout(resolve, 3000));

        // Send motion OFF
        console.log('[2/2] Sending motion OFF...');
        client.publish(topic, 'OFF');
        const offTimestamp = new Date().toISOString();
        console.log(`✓ Published motion OFF at ${offTimestamp}`);

        // Wait a moment before disconnecting
        await new Promise(resolve => setTimeout(resolve, 1000));

        console.log('\n✓ Motion cycle complete. Disconnecting...');
        client.end();
        process.exit(0);
    });

    client.on('error', (err) => {
        console.error('MQTT Error:', err);
        process.exit(1);
    });
}

// Run the script
main().catch(console.error);
