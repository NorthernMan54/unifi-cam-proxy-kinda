const mqtt = require('mqtt');
const fs = require('fs');

const config = JSON.parse(fs.readFileSync('config.json', 'utf8'));

// Create a manual event
async function createManualEvent(camera, label, duration = 30, confidence = 0.5) {
    const response = await fetch(`${config.baseUrl}/api/events/${camera}/${label}/create`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            duration: duration,  // Duration in seconds
            include_recording: true,
            score: confidence,   // Confidence score (0.0 to 1.0)
            draw: {
                box: [100, 100, 200, 200]  // Optional: [x1, y1, x2, y2]
            }
        })
    });

    const event = await response.json();
    console.log(`Event created: ${config.baseUrl}/api/events/${camera}/${label}/create <-`, JSON.stringify({
        duration: duration,  // Duration in seconds
        include_recording: true,
        score: confidence,   // Confidence score (0.0 to 1.0)
        draw: {
            box: [100, 100, 200, 200]  // Optional: [x1, y1, x2, y2]
        }
    }), ' -> ', event);
    return event;
}

// Example usage
createManualEvent(config.cameras[0], 'person', 30, 0.9);