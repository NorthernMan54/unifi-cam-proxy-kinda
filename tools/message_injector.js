const mqtt = require('mqtt');
const fs = require('fs');

const config = JSON.parse(fs.readFileSync('config.json', 'utf8'));

const sleep = (seconds) => new Promise(resolve => setTimeout(resolve, seconds * 1000));

// Promisify client.publish
function publishAsync(client, topic, payload) {
    return new Promise((resolve, reject) => {
        client.publish(topic, payload, (err) => {
            if (err) reject(err);
            else resolve();
        });
    });
}

async function main() {
    console.log('=== Frigate Zone Car Message ===');
    const messagesFile = process.argv[2] || "zone_car_messages.json";
    console.log(`Message file: ${messagesFile}`);
    const messages = JSON.parse(fs.readFileSync(messagesFile, 'utf8'));
    console.log(`Messages: ${messages.length}`);
    console.log(`Connecting to MQTT broker: ${config.mqttBroker}`);
    const topic = `${config.frigateMqttPrefix}/events`;
    console.log(`Publishing to topic: ${topic}\n`);
    // topic.matches(f"{prefix}/{camera}/motion"):
    const motionTopic = `${config.frigateMqttPrefix}/${config.frigateCamera}/motion`;

    const client = mqtt.connect(config.mqttBroker);

    await new Promise((resolve, reject) => {
        client.on('error', reject);
        client.on('connect', resolve);
    });

    console.log('✓ Connected to MQTT broker\n');

    await publishAsync(client, motionTopic, "ON");
    console.log(`✓ Published "ON" to ${motionTopic}`);
    await sleep(2); // Only sleep between messages, not after the last one   
    // Use a for...of loop so await actually works

    //    after = frigate_msg.get("after", {})
    //    descriptor = self.build_descriptor_from_frigate_msg(frigate_msg, object_type)
    //    frame_time_ms = int(after.get("frame_time", 0) * 1000) - self.args.frigate_time_sync_ms

    //    "frame_time": 1780844299.181006,

    for (const [index, message] of messages.entries()) {


        const now = Date.now();
        const hrtime = process.hrtime();
        const microseconds = (now % 1000) * 1000 + Math.floor(hrtime[1] / 1000) % 1000;
        const timestamp = Math.floor(now / 1000) + microseconds / 1000000;

        console.log(`Setting message ${index + 1} timestamp ${message.after.frame_time} to ${timestamp} (current time)`);
        message.after.frame_time = timestamp;

        console.log(`Publishing message ${index + 1}/${messages.length} to topic "${topic}"...`);
        var payload = JSON.stringify(message);

        console.log(`Payload: ${topic} - ${payload}\n`);

        await publishAsync(client, topic, payload);
        console.log(`✓ Message ${index + 1} published successfully`);

        await sleep(2); // Only sleep between messages, not after the last one
    }

    await publishAsync(client, motionTopic, "OFF");
    console.log(`✓ Published "OFF" to ${motionTopic}`);
    await sleep(2); // Only sleep between messages, not after the last one
    console.log('\n✓ All event lifecycles complete. Disconnecting...');
    client.end();
    process.exit(0);
}

if (require.main === module) {
    main().catch(console.error);
}
