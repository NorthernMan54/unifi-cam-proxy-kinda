const fs = require('fs');

const config = JSON.parse(fs.readFileSync('config.json', 'utf8'));
const baseUrl = config.baseUrl;

const params = new URLSearchParams({
    camera: config.cameras[0],
    limit: '15',
    has_snapshot: '1'
});

console.log(`Fetching recent events from ${baseUrl}/api/events?${params.toString()}`);

fetch(`${baseUrl}/api/events?${params}`)
    .then(response => response.json())
    .then(events => {
        // Save raw JSON response to file
        const outputFile = 'recent-events-response.json';
        fs.writeFileSync(outputFile, JSON.stringify(events, null, 2));
        console.log(`Raw JSON response saved to ${outputFile}\n`);

        events.forEach(event => {
            console.log(`Camera: ${event.camera}`);
            console.log(`Object: ${event.label}`);
            console.log(`Start: ${new Date(event.start_time * 1000).toLocaleString()}`);
            console.log(`Status: ${event.end_time === null ? 'Active' : 'Completed'}`);
            console.log(`Snapshot: ${baseUrl}/api/events/${event.id}/snapshot.jpg?bbox=1\n`);
            console.log(`Thumbnail: ${baseUrl}/api/events/${event.id}/thumbnail.jpg\n`);
        });
    })
    .catch(error => console.error('Error:', error));