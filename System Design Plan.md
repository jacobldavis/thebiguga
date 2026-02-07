# **Auditory Authentication System \- System Design Plan**

## **1\. Project Overview**

**Goal:** Create a secure, biometric-style authentication system where the "password" is a rhythmic and melodic sequence played by the user (e.g., on a virtual keyboard).  
**Core Challenge:** Unlike text passwords (exact match), musical input is analog and imperfect. The system must measure the *similarity* between a stored template and a live input, handling timing variations, polyphony (chords), and duration.

## **2\. Data Model (The "Piano Roll")**

We will not store audio files (WAV/MP3). We will store a lightweight, event-based JSON structure representing the "Piano Roll."

### **2.1 The Note Event Structure**

A single note press is defined by:

* **Pitch:** The identifier (MIDI note number or frequency string like "C4").  
* **Start Time:** Milliseconds relative to the *start of the sequence*.  
* **Duration:** How long the key was held in milliseconds.

### **2.2 The Password Payload**

A full password is a list of these events.  
\[  
  { "pitch": "C4", "start": 0,    "duration": 500 },  
  { "pitch": "E4", "start": 0,    "duration": 500 }, // Concurrent with C4 (Polyphony)  
  { "pitch": "G4", "start": 600,  "duration": 300 }  // Played after a 100ms pause  
\]

### **2.3 Frontend Capture Logic**

1. **Listen:** Attach event listeners to keydown (start) and keyup (end).  
2. **Record:** Store absolute timestamps using performance.now().  
3. **Normalize:** When the user clicks "Submit":  
   * Find the start\_time of the very first note.  
   * Subtract that value from *all* notes (so the sequence always starts at $t=0$).  
   * Send this normalized JSON to the backend.

## **3\. The Comparison Algorithm: Temporal Intersection over Union (IoU)**

Since we need a fuzzy match, we use **Intersection over Union**. This geometric approach naturally handles polyphony and timing errors.  
**Formula:**  
$$ Score \= \\frac{\\text{Total Time Overlap (Intersection)}}{\\text{Total Time Covered (Union)}} $$

### **3.1 Algorithm Steps (Python Backend)**

1. **Grouping:** Group both the *Stored Pattern* (Template) and *Input Pattern* (Attempt) by pitch.  
   * *Example:* All "C4" events go into one list, all "E4" into another.  
2. **Per-Pitch Calculation:** For each unique pitch found in either pattern:  
   * Calculate the 1D Union (total time the note is active in *either* pattern).  
   * Calculate the 1D Intersection (time the note is active in *both* patterns simultaneously).  
3. **Summation:**  
   * Total\_Intersection \= Sum of intersections across all pitches.  
   * Total\_Union \= Sum of unions across all pitches.  
4. **Final Score:**  
   * Similarity \= Total\_Intersection / Total\_Union  
   * Result is between 0.0 (complete mismatch) and 1.0 (perfect match).

### **3.2 Handling Tolerance (The "Fudge Factor")**

Humans are imperfect.

* **Buffer:** Before calculating overlap, expand the user's input duration by a small buffer (e.g., ±20ms) to allow for slight timing jitters.  
* **Threshold:** Define a strictness variable (e.g., 0.75). If Similarity \> 0.75, authentication succeeds.

## **4\. Backend Architecture (Python)**

### **4.1 Tech Stack**

* **Framework:** Flask or FastAPI (Lightweight, fast).  
* **Database:** PostgreSQL or SQLite (for prototyping).  
* **Math:** Native Python or numpy for interval calculations.

### **4.2 Database Schema**

Since the "password" is a complex object, we store it as a JSON blob.  
**Table: users**  
| Column | Type | Description |  
| :--- | :--- | :--- |  
| id | UUID | Primary Key |  
| username | String | Unique username |  
| password\_template | JSON/Blob | **Encrypted** JSON of the recorded pattern |  
| salt | String | Random salt used for encryption key derivation |  
| threshold | Float | (Optional) Custom strictness per user, default 0.8 |

### **4.3 Security Implementation**

**Crucial:** You cannot hash this password (because hashes require exact matches). You must store the reference template.

1. **Encryption:** Do not store raw JSON. Encrypt the password\_template field using a symmetric key (e.g., AES-256).  
2. **Key Management:** The decryption key should be an environment variable on the server, never stored in the DB.  
3. **Process:**  
   * User Login \-\> Server fetches Encrypted Blob \-\> Server Decrypts in RAM \-\> Runs Algorithm \-\> Clears RAM.

## **5\. API Endpoints**

### **POST /api/register**

* **Body:**  
  {  
    "username": "mozart\_22",  
    "pattern": \[ ...event\_list... \]  
  }

* **Logic:**  
  1. Validate input format.  
  2. Normalize timestamps (ensure first note is at 0).  
  3. Encrypt pattern.  
  4. Save to DB.

### **POST /api/login**

* **Body:**  
  {  
    "username": "mozart\_22",  
    "input\_pattern": \[ ...event\_list... \]  
  }

* **Logic:**  
  1. Fetch user record.  
  2. Decrypt stored pattern.  
  3. Normalize input\_pattern (shift to 0).  
  4. Run calculate\_similarity(stored, input).  
  5. If score \> threshold: Return 200 OK \+ JWT Token.  
  6. Else: Return 401 Unauthorized.

## **6\. Hackathon "Nice-to-Haves" (Stretch Goals)**

1. **Tempo Scaling:**  
   * If the user plays the correct rhythm but at 0.5x speed, standard IoU will fail.  
   * *Fix:* Calculate the total duration of Stored vs Input. If Input is 2x longer, scale all input timestamps by 0.5 before comparing.  
2. **Visual Feedback:**  
   * On failure, show the user a visual overlay of their attempt vs. the stored pattern so they can see *where* they missed (too early/too late).  
3. **MIDI Support:**  
   * Allow users to plug in a real MIDI keyboard via the Web MIDI API instead of using the computer keyboard.