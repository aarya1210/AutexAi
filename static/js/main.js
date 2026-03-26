document.addEventListener('DOMContentLoaded', function () {
  // Auto-dismiss alerts
  document.querySelectorAll('.alert').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity .4s';
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 400);
    }, 6000);
  });

  // Questionnaire progress
  const form = document.getElementById('qForm');
  if (form) {
    const fill = document.getElementById('progressFill');
    const txt  = document.getElementById('progressText');
    const cards = document.querySelectorAll('.q-card');
    const total = cards.length;

    function updateProgress() {
      let done = 0;
      const groups = {};
      form.querySelectorAll('input[type="radio"]').forEach(r => {
        groups[r.name] = groups[r.name] || false;
        if (r.checked) groups[r.name] = true;
      });
      done = Object.values(groups).filter(Boolean).length;
      const pct = Math.round((done / total) * 100);
      if (fill) fill.style.width = pct + '%';
      if (txt)  txt.textContent = `${done} / ${total}`;
    }

    form.querySelectorAll('input[type="radio"]').forEach(r => {
      r.addEventListener('change', function() {
        updateProgress();
        // Smooth scroll to next question
        const currentCard = this.closest('.q-card');
        const nextCard = currentCard.nextElementSibling;
        if (nextCard && nextCard.classList.contains('q-card')) {
          setTimeout(() => {
            nextCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }, 300);
        }
      });
    });

    form.addEventListener('submit', function (e) {
      const groups = {};
      form.querySelectorAll('input[type="radio"]').forEach(r => {
        groups[r.name] = groups[r.name] || false;
        if (r.checked) groups[r.name] = true;
      });
      const unanswered = Object.entries(groups).find(([, answered]) => !answered);
      if (unanswered) {
        e.preventDefault();
        showToast('Please answer all questions', 'error');
        const firstUnanswered = form.querySelector(`input[name="${unanswered[0]}"]`);
        if (firstUnanswered) firstUnanswered.closest('.q-card').scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
      }
      const btn = document.getElementById('submitBtn');
      if (btn) {
        btn.innerHTML = '⏳ Analyzing...';
        btn.disabled = true;
        btn.style.opacity = '.7';
      }
    });
  }

  // Password match
  const pwd  = document.querySelector('input[name="password"]');
  const cpwd = document.querySelector('input[name="confirm_password"]');
  if (pwd && cpwd) {
    cpwd.addEventListener('input', function () {
      cpwd.setCustomValidity(cpwd.value !== pwd.value ? 'Passwords do not match' : '');
    });
  }

  // Animate gauge
  const gaugeFill = document.querySelector('.gauge-fill');
  if (gaugeFill) {
    const target = gaugeFill.style.width;
    gaugeFill.style.width = '0%';
    setTimeout(() => {
      gaugeFill.style.transition = 'width 1.5s ease';
      gaugeFill.style.width = target;
    }, 300);
  }

  // Animate donuts
  document.querySelectorAll('.big-donut').forEach(el => {
    el.style.transform = 'scale(0)';
    el.style.transition = 'transform .6s cubic-bezier(.34,1.56,.64,1)';
    setTimeout(() => { el.style.transform = 'scale(1)'; }, 200);
  });

  // Toast helper
  function showToast(msg, type) {
    const div = document.createElement('div');
    div.className = `alert alert-${type === 'error' ? 'error' : 'info'}`;
    div.style.cssText = 'position:fixed;top:80px;right:20px;z-index:9999;min-width:300px;box-shadow:0 6px 20px rgba(0,0,0,.2);animation:slideIn .4s ease;';
    div.innerHTML = msg + '<button onclick="this.parentElement.remove()">×</button>';
    document.body.appendChild(div);
    setTimeout(() => div.remove(), 5000);
  }
});
