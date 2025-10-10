// script.js

const API_BASE_URL = "http://127.0.0.1:8000"; // Default FastAPI Uvicorn address
const COLORS = ["Blue", "Yellow", "White", "Green", "Red"];
let selectedCardIndex = null;
let selectedDrawSource = null;
let gameData = null;

// --- Rendering Functions ---

function getCardHTML(card, isHand = false) {
    if (!card) return '<span class="card">EMPTY</span>';
    const [color_idx, value] = card;
    const colorName = COLORS[color_idx];
    const cardText = value === 0 ? 'X' : value;
    let html = `<span class="card color-${colorName}" data-color="${colorName}" data-value="${value}">`;
    html += cardText;
    html += '</span>';
    return html;
}

function renderExpeditions(playerId, expeditions) {
    const container = document.getElementById(`player-${playerId}-expeditions`);
    container.innerHTML = `<h2>Player ${playerId} Expeditions</h2>`;
    
    expeditions[playerId].forEach((expedition, color_idx) => {
        const colorName = COLORS[color_idx];
        const cardsHTML = expedition.map(card => getCardHTML(card)).join('');
        const expDiv = document.createElement('div');
        expDiv.className = 'expedition-row';
        expDiv.innerHTML = `<strong>${colorName}:</strong> ${cardsHTML || 'Start'}`;
        container.appendChild(expDiv);
    });
}

function renderDiscardPiles(state) {
    const container = document.getElementById('discard-piles');
    container.innerHTML = '';
    
    state.discard_piles.forEach((top_card, color_idx) => {
        const colorName = COLORS[color_idx];
        const cardHTML = getCardHTML(top_card);
        const pileDiv = document.createElement('div');
        pileDiv.innerHTML = `<strong>${colorName}:</strong> ${cardHTML}`;
        container.appendChild(pileDiv);
    });
}

function renderDrawOptions(state) {
    const container = document.getElementById('draw-options');
    container.innerHTML = '';

    // Option 0: Draw from Deck
    let deckOption = document.createElement('span');
    deckOption.className = 'draw-option';
    deckOption.textContent = `Deck (${state.deck_size})`;
    deckOption.dataset.source = 0;
    deckOption.onclick = () => selectDrawSource(0);
    container.appendChild(deckOption);

    // Options 1-5: Draw from Discard Piles
    state.discard_piles.forEach((top_card, color_idx) => {
        const source = color_idx + 1;
        let discardOption = document.createElement('span');
        discardOption.className = 'draw-option';
        discardOption.dataset.source = source;
        discardOption.textContent = `${COLORS[color_idx]} Discard (Top: ${top_card ? top_card[1] : '?'})`;
        
        if (state.full_discard_piles[color_idx].length > 0) {
            discardOption.onclick = () => selectDrawSource(source);
        } else {
            discardOption.style.opacity = 0.5;
            discardOption.title = "Pile is empty";
        }
        container.appendChild(discardOption);
    });
    
    // Reselect if current draw source is still valid
    if (selectedDrawSource !== null) {
        selectDrawSource(selectedDrawSource);
    }
}

function renderHand(hand) {
    const container = document.getElementById('player-hand');
    container.innerHTML = '';
    
    hand.forEach((card, index) => {
        const cardDiv = document.createElement('span');
        cardDiv.className = 'card hand-card ' + `color-${COLORS[card[0]]}`;
        cardDiv.textContent = card[1] === 0 ? 'X' : card[1];
        cardDiv.dataset.index = index;
        cardDiv.dataset.color = card[0];
        cardDiv.dataset.value = card[1];
        cardDiv.onclick = () => selectCard(index);
        container.appendChild(cardDiv);
    });
    
    // Re-select if card index is still valid
    if (selectedCardIndex !== null && selectedCardIndex < hand.length) {
        selectCard(selectedCardIndex);
    } else {
        selectedCardIndex = null;
        updateActionButtons();
    }
}

function updateStatus(state) {
    document.getElementById('current-player').textContent = `P${state.current_player}`;
    document.getElementById('deck-size').textContent = state.deck_size;
    document.getElementById('score-p0').textContent = state.scores[0].toFixed(1);
    document.getElementById('score-p1').textContent = state.scores[1].toFixed(1);
    document.getElementById('game-message').textContent = state.game_over ? "GAME OVER!" : "";
    
    const viewingPlayer = parseInt(document.getElementById('player-select').value);
    document.getElementById('current-view-player').textContent = viewingPlayer;
    
    // Only enable controls for the current player when they are the one viewing
    const isPlayersTurn = state.current_player === viewingPlayer;
    document.getElementById('play-expedition-btn').disabled = !isPlayersTurn;
    document.getElementById('play-discard-btn').disabled = !isPlayersTurn;
}

// --- Interaction Functions ---

function selectCard(index) {
    selectedCardIndex = index;
    document.querySelectorAll('.hand-card').forEach(card => {
        card.classList.remove('selected');
        if (parseInt(card.dataset.index) === index) {
            card.classList.add('selected');
        }
    });
    updateActionButtons();
}

function selectDrawSource(source) {
    selectedDrawSource = source;
    document.querySelectorAll('.draw-option').forEach(opt => {
        opt.classList.remove('selected');
        if (parseInt(opt.dataset.source) === source) {
            opt.classList.add('selected');
        }
    });
    updateActionButtons();
}

function updateActionButtons() {
    const playExpeditionBtn = document.getElementById('play-expedition-btn');
    const playDiscardBtn = document.getElementById('play-discard-btn');
    const isReady = selectedCardIndex !== null && selectedDrawSource !== null;
    
    // Enable/Disable based on readiness
    playExpeditionBtn.disabled = !isReady;
    playDiscardBtn.disabled = !isReady;
    
    // Add logic to check VALIDITY based on game state (requires fetching valid_actions)
    // For simplicity, we skip server-side validation here, relying on the backend to reject invalid moves.
}

async function submitAction(actionType) {
    if (selectedCardIndex === null || selectedDrawSource === null || gameData.game_over) {
        alert("Please select a card and a draw source.");
        return;
    }

    const playerId = parseInt(document.getElementById('player-select').value);
    
    const payload = {
        player_id: playerId,
        card_index: selectedCardIndex,
        action_type: actionType,
        draw_source: selectedDrawSource
    };

    try {
        const response = await fetch(`${API_BASE_URL}/game/play`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        
        if (!response.ok) {
            document.getElementById('game-message').textContent = `Error: ${data.detail}`;
            return;
        }

        document.getElementById('game-message').textContent = data.message;
        
        // Update game state immediately
        gameData = data.state;
        renderGame(gameData);
        
        // Clear selection after a successful move
        selectedCardIndex = null;
        selectedDrawSource = null;
        document.querySelectorAll('.selected').forEach(el => el.classList.remove('selected'));
        updateActionButtons();

    } catch (error) {
        document.getElementById('game-message').textContent = `Network Error: ${error.message}`;
        console.error('Error submitting action:', error);
    }
}

// --- Main Flow ---

async function loadGameState() {
    const playerId = parseInt(document.getElementById('player-select').value);
    try {
        const response = await fetch(`${API_BASE_URL}/game/state/${playerId}`);
        if (!response.ok) throw new Error("Failed to fetch game state.");
        
        gameData = await response.json();
        renderGame(gameData);
    } catch (error) {
        document.getElementById('game-message').textContent = `Could not connect to server or load game state. ${error.message}`;
        console.error('Error loading game state:', error);
    }
}

function renderGame(state) {
    const viewingPlayer = parseInt(document.getElementById('player-select').value);
    
    // Render the board
    renderExpeditions(0, state.expeditions);
    renderExpeditions(1, state.expeditions);
    renderDiscardPiles(state);
    
    // Render the viewing player's private information
    document.getElementById('player-hand').innerHTML = ''; // Clear if not viewing hand
    if (viewingPlayer === state.player_id) {
        renderHand(state.hand);
    } else {
        document.getElementById('player-hand').textContent = `Viewing P${viewingPlayer} hand. (Hand size: ${state.opponent_hand_size})`;
    }

    // Render global status and draw options
    renderDrawOptions(state);
    updateStatus(state);
}

async function resetGame() {
    try {
        const response = await fetch(`${API_BASE_URL}/game/reset`);
        if (!response.ok) throw new Error("Failed to reset game.");
        
        document.getElementById('game-message').textContent = "Game reset. Dealing cards...";
        selectedCardIndex = null;
        selectedDrawSource = null;
        await loadGameState();
    } catch (error) {
        document.getElementById('game-message').textContent = `Error resetting game: ${error.message}`;
    }
}

async function setOpponent() {
    opponent = allOpponents[selectElement.value].name;
     try {
        const payload = {path : opponent}
        const response = await fetch(`${API_BASE_URL}/game/set_opponent`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        
        if (!response.ok) {
            document.getElementById('game-message').textContent = `Error: ${data.detail}`;
            return;
        }
        await resetGame();
    } catch (error) {
        document.getElementById('game-message').textContent = `Network Error: ${error.message}`;
        console.error('Error setting opponent:', error);
    }
}

// Use a flag to ensure the data is only fetched once per session
let playersLoaded = false;
const selectElement = document.getElementById('opponent-select');
const statusElement = document.getElementById('status-message');
const statusTextElement = document.getElementById('status-text');
let allOpponents = [];
/**
 * Simulates an asynchronous server call to fetch player data.
 * @returns {Promise<Array<{id: string, name: string}>>} A promise that resolves with player data.
 */
async function fetchPlayersFromServer() {
    return (await fetch(`${API_BASE_URL}/game/opponent`)).json();
}

/**
 * Updates the <select> element with fetched player options.
 * It preserves the initial static options and adds the dynamic ones.
 * @param {Array<{id: string, name: string}>} players - The list of players from the server.
 */
function updatePlayerSelect(players) {
    // Clear existing options (except the first few, if desired, but here we just rebuild)
    // It's often safer to rebuild entirely or remove placeholders.
    selectElement.innerHTML = '';

    // Add the dynamically loaded players
    players.forEach(player => {
        const option = document.createElement('option');
        option.value = player.id;
        option.textContent = player.name;
        selectElement.appendChild(option);
    });

    selectElement.value = '';
}

/**
 * Called when the user focuses on the select element (i.e., clicks it).
 * Fetches data only if it hasn't been loaded yet.
 */
async function fetchPlayersIfNecessary() {
    // if (playersLoaded) {
    //     console.log("Players already loaded. Skipping fetch.");
    //     return;
    // }
    
    // Set temporary loading state in the UI
    statusElement.classList.remove('hidden');
    statusElement.classList.add('bg-yellow-50', 'border-yellow-400', 'text-yellow-800');
    statusElement.classList.remove('bg-indigo-50', 'border-indigo-400', 'text-indigo-800');
    statusTextElement.textContent = 'Fetching player list from the server...';
    
    // Change the temporary text inside the select box
    const initialOptions = selectElement.innerHTML;
    selectElement.innerHTML = '<option value="" disabled selected>Fetching...</option>';
    
    try {
        const players = await fetchPlayersFromServer();
        allOpponents = players;
        updatePlayerSelect(players);
        playersLoaded = true; // Set the flag after successful load

        statusElement.classList.add('bg-green-50', 'border-green-400', 'text-green-800');
        statusElement.classList.remove('bg-yellow-50', 'border-yellow-400', 'text-yellow-800');
        statusTextElement.textContent = `Successfully loaded ${players.length} online players.`;

    } catch (error) {
        console.error('Error fetching players:', error);
        statusElement.classList.add('bg-red-50', 'border-red-400', 'text-red-800');
        statusElement.classList.remove('bg-yellow-50', 'border-yellow-400', 'text-yellow-800');
        statusTextElement.textContent = 'Failed to load players. Please try again later.';
        
        // Restore initial options if fetch fails
        selectElement.innerHTML = initialOptions;
        selectElement.value = ''; // Ensure nothing is selected
    }
}


// Initial load
document.addEventListener('DOMContentLoaded', () => {
    loadGameState();
});