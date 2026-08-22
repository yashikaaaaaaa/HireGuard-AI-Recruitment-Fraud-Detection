// ============================================================
// HIREGUARD AI — FRONTEND APPLICATION
// ============================================================

const API_URL = "http://127.0.0.1:8000/predict";

const jobText = document.getElementById("jobText");
const analyzeButton = document.getElementById("analyzeButton");
const clearButton = document.getElementById("clearButton");

const characterCount = document.getElementById("characterCount");

const loader = document.getElementById("loader");

const resultCard = document.getElementById("resultCard");
const predictionText = document.getElementById("predictionText");
const decisionScore = document.getElementById("decisionScore");
const modelName = document.getElementById("modelName");
const resultIcon = document.getElementById("resultIcon");
const riskMessage = document.getElementById("riskMessage");

const errorMessage = document.getElementById("errorMessage");


// ============================================================
// CHARACTER COUNTER
// ============================================================

jobText.addEventListener("input", () => {

    const count = jobText.value.length;

    characterCount.textContent =
        `${count.toLocaleString()} characters`;

});


// ============================================================
// CLEAR BUTTON
// ============================================================

clearButton.addEventListener("click", () => {

    jobText.value = "";

    characterCount.textContent =
        "0 characters";

    resultCard.classList.add("hidden");

    errorMessage.classList.add("hidden");

    jobText.focus();

});


// ============================================================
// ANALYZE JOB
// ============================================================

analyzeButton.addEventListener(
    "click",
    analyzeJob
);


async function analyzeJob() {

    const text = jobText.value.trim();

    // --------------------------------------------------------
    // Input validation
    // --------------------------------------------------------

    if (!text) {

        showError(
            "Please paste a job advertisement before running the analysis."
        );

        return;
    }


    // --------------------------------------------------------
    // Reset interface
    // --------------------------------------------------------

    errorMessage.classList.add("hidden");

    resultCard.classList.add("hidden");

    loader.classList.remove("hidden");

    analyzeButton.disabled = true;

    analyzeButton.innerHTML =
        "<span>Analyzing...</span>";


    try {

        // ----------------------------------------------------
        // API REQUEST
        // ----------------------------------------------------

        const response = await fetch(
            API_URL,
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    text: text
                })
            }
        );


        if (!response.ok) {

            throw new Error(
                `API returned status ${response.status}`
            );

        }


        const data = await response.json();


        // ----------------------------------------------------
        // DISPLAY RESULT
        // ----------------------------------------------------

        displayResult(data);

    }

    catch (error) {

        console.error(
            "HireGuard API Error:",
            error
        );


        showError(
            "Unable to connect to the HireGuard AI backend. Make sure the FastAPI server is running."
        );

    }

    finally {

        loader.classList.add("hidden");

        analyzeButton.disabled = false;

        analyzeButton.innerHTML =
            "<span>Analyze Job Posting</span><span>→</span>";

    }

}


// ============================================================
// DISPLAY PREDICTION
// ============================================================

function displayResult(data) {

    resultCard.classList.remove(
        "hidden",
        "legitimate",
        "fraudulent"
    );


    predictionText.textContent =
        data.prediction;

    decisionScore.textContent =
        Number(data.decision_score).toFixed(4);

    modelName.textContent =
        data.model || "Linear SVM";


    // --------------------------------------------------------
    // FRAUDULENT
    // --------------------------------------------------------

    if (
        data.prediction ===
        "Potentially Fraudulent"
    ) {

        resultCard.classList.add(
            "fraudulent"
        );

        resultIcon.textContent = "!";

        riskMessage.innerHTML = `
            <strong>Potential recruitment risk detected.</strong>
            The advertisement contains textual patterns associated
            with fraudulent recruitment postings. Verify the company,
            recruiter identity and application process before sharing
            personal information or making any payment.
        `;

    }

    // --------------------------------------------------------
    // LEGITIMATE
    // --------------------------------------------------------

    else {

        resultCard.classList.add(
            "legitimate"
        );

        resultIcon.textContent = "✓";

        riskMessage.innerHTML = `
            <strong>No strong fraud pattern detected.</strong>
            The advertisement is classified as likely legitimate
            based on patterns learned by the model. You should still
            verify the employer and recruitment source before applying.
        `;

    }


    resultCard.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });

}


// ============================================================
// ERROR HANDLING
// ============================================================

function showError(message) {

    errorMessage.textContent =
        message;

    errorMessage.classList.remove(
        "hidden"
    );

}