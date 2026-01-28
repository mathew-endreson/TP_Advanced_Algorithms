document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('runBtn');
    const numItemsInput = document.getElementById('numItems');
    const capacityInput = document.getElementById('capacity');
    const resultsBody = document.getElementById('resultsBody');

    // Algorithm order matching the backend returns or predefined list
    // Actually backend returns a list, we can just clear and repopulate

    runBtn.addEventListener('click', async () => {
        const numItems = parseInt(numItemsInput.value);
        const capacity = parseInt(capacityInput.value);

        if (!numItems || !capacity) return; // Simple validation

        // Set Loading State
        setLoadingState(true);
        runBtn.disabled = true;

        console.log("Sending request to /benchmark");
        try {
            const response = await fetch('/benchmark', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ num_items: numItems, capacity: capacity })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            updateResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred: ' + error.message);
        } finally {
            setLoadingState(false);
            runBtn.disabled = false;
        }
    });

    function setLoadingState(isLoading) {
        const dots = document.querySelectorAll('.status-dot');
        dots.forEach(dot => {
            if (isLoading) {
                dot.dataset.originalClass = dot.className;
                dot.className = 'status-dot dot-loading';
            } else {
                // We will replace the whole table body anyway, but for completeness
            }
        });

        // visual feedback on button
        if (isLoading) {
            runBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
        } else {
            runBtn.innerHTML = '<i class="fa-solid fa-rocket"></i>';
        }
    }

    function updateResults(data) {
        resultsBody.innerHTML = ''; // Clear current rows

        data.forEach(algo => {
            const row = document.createElement('tr');

            // Map status color class from tailwind text-color to our css dot classes if needed
            // Backend sends 'text-orange-500', we need 'dot-orange'
            // Let's create a mapper or just use the color from the name
            let dotClass = 'dot-green'; // default
            if (algo.status_color.includes('orange')) dotClass = 'dot-orange';
            else if (algo.status_color.includes('teal')) dotClass = 'dot-teal';
            else if (algo.status_color.includes('purple')) dotClass = 'dot-purple';
            else if (algo.status_color.includes('red')) dotClass = 'dot-red';

            row.innerHTML = `
                <td>${algo.name}</td>
                <td>${algo.value}</td>
                <td>${algo.time_ms} ms</td>
                <td><span class="status-dot ${dotClass}"></span></td>
            `;
            resultsBody.appendChild(row);
        });
    }
});
