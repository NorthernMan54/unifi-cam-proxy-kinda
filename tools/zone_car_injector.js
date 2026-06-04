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

	const client = mqtt.connect(config.mqttBroker);

	await new Promise((resolve, reject) => {
		client.on('error', reject);
		client.on('connect', resolve);
	});

	console.log('✓ Connected to MQTT broker\n');

	// Use a for...of loop so await actually works
	for (const [index, message] of messages.entries()) {
		console.log(`Publishing message ${index + 1}/${messages.length} to topic "${topic}"...`);
		const payload = JSON.stringify(message);
		console.log(`Payload: ${topic} - ${payload}\n`);

		await publishAsync(client, topic, payload);
		console.log(`✓ Message ${index + 1} published successfully`);

		await sleep(2); // Only sleep between messages, not after the last one
	}

	console.log('\n✓ All event lifecycles complete. Disconnecting...');
	client.end();
	process.exit(0);
}

if (require.main === module) {
	main().catch(console.error);
}