const mqtt = require('mqtt');
const fs = require('fs');

// Read configuration
const config = JSON.parse(fs.readFileSync('config.json', 'utf8'));

// Configuration - Using Frigate's camera viewport
const CAMERA_WIDTH = 1280;
const CAMERA_HEIGHT = 720;
const BOX_SIZE = 50;

// Position definitions (in camera viewport coordinates)
const POSITIONS = {
    topLeft: { x: 0, y: 0 },
    topRight: { x: CAMERA_WIDTH - BOX_SIZE, y: 0 },
    bottomLeft: { x: 0, y: CAMERA_HEIGHT - BOX_SIZE },
    bottomRight: { x: CAMERA_WIDTH - BOX_SIZE, y: CAMERA_HEIGHT - BOX_SIZE },
    center: { x: (CAMERA_WIDTH - BOX_SIZE) / 2, y: (CAMERA_HEIGHT - BOX_SIZE) / 2 }
};

// Generate random event ID (Frigate format: timestamp.microseconds-random)
function generateEventId() {
    const now = Date.now() / 1000;
    const random = Math.random().toString(36).substring(2, 8);
    return `${now.toFixed(6)}-${random}`;
}

// Calculate bounding box from position
function calculateBox(position) {
    const pos = POSITIONS[position];
    if (!pos) {
        throw new Error(`Invalid position: ${position}. Valid options: ${Object.keys(POSITIONS).join(', ')}`);
    }

    const x1 = pos.x;
    const y1 = pos.y;
    const x2 = x1 + BOX_SIZE;
    const y2 = y1 + BOX_SIZE;

    return [
        Math.round(x1),
        Math.round(y1),
        Math.round(x2),
        Math.round(y2)
    ];
}

// Calculate region around the box (slightly larger)
function calculateRegion(box) {
    const margin = 50;
    return [
        Math.max(0, box[0] - margin),
        Math.max(0, box[1] - margin),
        Math.min(CAMERA_WIDTH, box[2] + margin),
        Math.min(CAMERA_HEIGHT, box[3] + margin)
    ];
}

// Calculate normalized path data based on box position
function calculatePathData(box, baseTime) {
    const centerX = (box[0] + box[2]) / 2 / CAMERA_WIDTH;
    const centerY = (box[1] + box[3]) / 2 / CAMERA_HEIGHT;

    return [
        [[centerX, centerY], baseTime - 2],
        [[centerX, centerY], baseTime - 1],
        [[centerX, centerY], baseTime]
    ];
}

// Create Frigate event payload
function createEventPayload(type, eventId, camera, box, isEnd = false) {
    const baseTime = Date.now() / 1000;
    const startTime = baseTime - 5;
    const area = BOX_SIZE * BOX_SIZE;
    const region = calculateRegion(box);
    const pathData = calculatePathData(box, baseTime);

    const snapshot = {
        frame_time: baseTime - 0.5,
        box: box,
        area: area,
        region: region,
        score: 0.9,
        attributes: [],
        current_estimated_speed: 0,
        velocity_angle: 0,
        path_data: pathData,
        recognized_license_plate: null,
        recognized_license_plate_score: null
    };

    const eventData = {
        id: eventId,
        camera: camera,
        frame_time: baseTime,
        snapshot: snapshot,
        label: "person",
        sub_label: null,
        top_score: 0.9,
        false_positive: false,
        start_time: startTime,
        end_time: isEnd ? baseTime : null,
        score: 0.9,
        box: box,
        area: area,
        ratio: 1.0,
        region: region,
        active: !isEnd,
        stationary: false,
        motionless_count: 0,
        position_changes: 1,
        current_zones: [],
        entered_zones: [],
        has_clip: isEnd,
        has_snapshot: true,
        attributes: {},
        current_attributes: [],
        pending_loitering: false,
        max_severity: "alert",
        current_estimated_speed: 0,
        average_estimated_speed: 0,
        velocity_angle: 0,
        path_data: pathData,
        recognized_license_plate: null
    };

    const payload = {
        before: JSON.parse(JSON.stringify(eventData)), // deep clone
        after: eventData,
        type: type
    };

    // Adjust before state slightly
    payload.before.frame_time = baseTime - 1;
    payload.before.snapshot.frame_time = baseTime - 1.5;

    return payload;
}

// Inject event lifecycle for a specific position
async function injectEventLifecycle(client, camera, position) {
    const box = calculateBox(position);
    const eventId = generateEventId();

    console.log(`\n--- Event Lifecycle for position: ${position} ---`);
    console.log(`Event ID: ${eventId}`);
    console.log(`Bounding box: [${box.join(', ')}] (${BOX_SIZE}x${BOX_SIZE} pixels)`);

    // Send "new" event
    console.log('[1/3] Sending "new" event... to', camera);
    const newEvent = createEventPayload('new', eventId, camera, box, false);
    client.publish(config.frigateTopic, JSON.stringify(newEvent));
    const newTimestamp = new Date().toISOString();
    console.log(`✓ Published "new" event - box: [${box.join(', ')}] at ${newTimestamp}`);

    await new Promise(resolve => setTimeout(resolve, 2000));

    // Send "update" event
    console.log('[2/3] Sending "update" event... to', camera);
    const updateEvent = createEventPayload('update', eventId, camera, box, false);
    client.publish(config.frigateTopic, JSON.stringify(updateEvent));
    const updateTimestamp = new Date().toISOString();
    console.log(`✓ Published "update" event - box: [${box.join(', ')}] at ${updateTimestamp}`);

    await new Promise(resolve => setTimeout(resolve, 2000));

    // Send "end" event
    console.log('[3/3] Sending "end" event... to', camera);
    const endEvent = createEventPayload('end', eventId, camera, box, true);
    client.publish(config.frigateTopic, JSON.stringify(endEvent));
    const endTimestamp = new Date().toISOString();
    console.log(`✓ Published "end" event - box: [${box.join(', ')}] at ${endTimestamp}`);

    await new Promise(resolve => setTimeout(resolve, 1000));
}

// Main function
async function main() {
    // Parse command line arguments
    const positions = process.argv.slice(2);

    if (positions.length === 0) {
        console.log('Usage: node unifi-injector.js <position1> [position2] ...');
        console.log(`Valid positions: ${Object.keys(POSITIONS).join(', ')}`);
        console.log('\nExample: node unifi-injector.js topRight bottomRight');
        console.log(`\nCamera resolution: ${CAMERA_WIDTH}x${CAMERA_HEIGHT}`);
        console.log(`Box size: ${BOX_SIZE}x${BOX_SIZE} pixels`);
        process.exit(1);
    }

    // Validate positions
    for (const pos of positions) {
        if (!POSITIONS[pos]) {
            console.error(`Error: Invalid position "${pos}"`);
            console.log(`Valid positions: ${Object.keys(POSITIONS).join(', ')}`);
            process.exit(1);
        }
    }

    console.log('=== Frigate MQTT Event Injector (Camera Viewport) ===');
    console.log(`Camera viewport: ${CAMERA_WIDTH}x${CAMERA_HEIGHT}`);
    console.log(`Box size: ${BOX_SIZE}x${BOX_SIZE} pixels`);
    console.log(`Positions to test: ${positions.join(', ')}`);
    console.log(`\nConnecting to MQTT broker: ${config.mqttBroker}`);

    const client = mqtt.connect(config.mqttBroker);

    client.on('connect', async () => {
        console.log('✓ Connected to MQTT broker\n');

        const camera = config.cameras[0];

        // Loop through each position
        for (let i = 0; i < positions.length; i++) {
            await injectEventLifecycle(client, camera, positions[i]);

            // Add 10 second delay between positions (except after the last one)
            if (i < positions.length - 1) {
                console.log('\n⏳ Waiting 10 seconds before next position...\n');
                await new Promise(resolve => setTimeout(resolve, 10000));
            }
        }

        console.log('\n✓ All event lifecycles complete. Disconnecting...');
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