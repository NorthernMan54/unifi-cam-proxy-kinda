const mqtt = require('mqtt');
const fs = require('fs');
const { POSITIONS, CAMERA_WIDTH, CAMERA_HEIGHT, BOX_SIZE, injectEventLifecycle } = require('./unifi-injector');

// Read configuration
const config = JSON.parse(fs.readFileSync('config.json', 'utf8'));

// Send motion ON event
async function sendMotionOn(client, camera) {
    const topic = `frigate/${camera}/motion`;
    console.log('\n🟢 [MOTION] Sending motion ON...');
    client.publish(topic, 'ON');
    const timestamp = new Date().toISOString();
    console.log(`✓ Published motion ON at ${timestamp}`);
}

// Send motion OFF event
async function sendMotionOff(client, camera) {
    const topic = `frigate/${camera}/motion`;
    console.log('\n🔴 [MOTION] Sending motion OFF...');
    client.publish(topic, 'OFF');
    const timestamp = new Date().toISOString();
    console.log(`✓ Published motion OFF at ${timestamp}`);
}

// Main function
async function main() {
    // Parse command line arguments
    const args = process.argv.slice(2);

    if (args.length === 0) {
        console.log('Usage: node combined-injector.js [camera] <position1> [position2] ...');
        console.log(`\nPositions: ${Object.keys(POSITIONS).join(', ')}`);
        console.log('\nExamples:');
        console.log('  node combined-injector.js topRight');
        console.log('  node combined-injector.js center bottomRight');
        console.log('  node combined-injector.js driveway topLeft center');
        console.log(`\nCamera resolution: ${CAMERA_WIDTH}x${CAMERA_HEIGHT}`);
        console.log(`Box size: ${BOX_SIZE}x${BOX_SIZE} pixels`);
        process.exit(1);
    }

    // Determine if first arg is a camera or position
    let camera;
    let positions;

    if (POSITIONS[args[0]]) {
        // First arg is a position, use default camera
        camera = config.cameras[0];
        positions = args;
    } else {
        // First arg is camera name
        camera = args[0];
        positions = args.slice(1);

        if (positions.length === 0) {
            console.error('Error: At least one position is required');
            console.log(`Valid positions: ${Object.keys(POSITIONS).join(', ')}`);
            process.exit(1);
        }
    }

    // Validate positions
    for (const pos of positions) {
        if (!POSITIONS[pos]) {
            console.error(`Error: Invalid position "${pos}"`);
            console.log(`Valid positions: ${Object.keys(POSITIONS).join(', ')}`);
            process.exit(1);
        }
    }

    console.log('=== Combined Motion + Unifi Event Injector ===');
    console.log(`Camera: ${camera}`);
    console.log(`Camera viewport: ${CAMERA_WIDTH}x${CAMERA_HEIGHT}`);
    console.log(`Box size: ${BOX_SIZE}x${BOX_SIZE} pixels`);
    console.log(`Positions to test: ${positions.join(', ')}`);
    console.log(`\nConnecting to MQTT broker: ${config.mqttBroker}`);

    const client = mqtt.connect(config.mqttBroker);

    client.on('connect', async () => {
        console.log('✓ Connected to MQTT broker\n');

        // Step 1: Send motion ON
        await sendMotionOn(client, camera);

        // Wait a moment after motion starts
        console.log('\n⏳ Waiting 2 seconds...');
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Step 2: Loop through each position and inject unifi events
        for (let i = 0; i < positions.length; i++) {
            await injectEventLifecycle(client, camera, positions[i]);

            // Add delay between positions (except after the last one)
            if (i < positions.length - 1) {
                console.log('\n⏳ Waiting 5 seconds before next position...\n');
                await new Promise(resolve => setTimeout(resolve, 5000));
            }
        }

        // Wait a moment before ending motion
        console.log('\n⏳ Waiting 2 seconds...');
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Step 3: Send motion OFF
        await sendMotionOff(client, camera);

        // Wait a moment before disconnecting
        await new Promise(resolve => setTimeout(resolve, 1000));

        console.log('\n✓ Complete cycle finished. Disconnecting...');
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
