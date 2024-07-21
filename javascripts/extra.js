document$.subscribe(function () {
    // // Check if the element with ID 'particles-js' exists
    // var particlesDiv = document.getElementById('particles-js');

    // // If it doesn't exist, create and append it to the body
    // if (!particlesDiv) {
    //     particlesDiv = document.createElement('div');
    //     particlesDiv.id = 'particles-js';
    //     document.body.appendChild(particlesDiv);
    // }

    // console.log("Initialize third-party libraries here")
    // particlesJS.load('particles-js', '/assets/particles.json', function () {
    //         console.log('callback - particles.js config loaded');
    // });

    // Check if the element with ID 'particles-js' exists
    var swarm_canvas = document.getElementById('swarm');

    // If it doesn't exist, create and append it to the body
    if (!swarm_canvas) {
        swarm_canvas = document.createElement('canvas');
        swarm_canvas.width = document.documentElement.clientWidth
        swarm_canvas.height = document.documentElement.clientHeight
        swarm_canvas.id = 'swarm';
        document.body.appendChild(swarm_canvas);
    }
})
