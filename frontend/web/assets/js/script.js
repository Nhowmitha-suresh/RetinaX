// Active navigation scroll spy & Scroll-to-Top button listener
window.addEventListener('scroll', () => {
  const sections = document.querySelectorAll('section');
  const navLinks = document.querySelectorAll('.nav-link');
  const scrollTopBtn = document.getElementById('scrollTopBtn');
  let current = 'home';

  sections.forEach(section => {
    const sectionTop = section.offsetTop - 150;
    if (window.scrollY >= sectionTop) {
      const id = section.getAttribute('id');
      if (id) current = id;
    }
  });

  navLinks.forEach(link => {
    link.classList.remove('active');
    if (link.getAttribute('href') === `#${current}`) {
      link.classList.add('active');
    }
  });

  // Toggle Scroll-to-Top Button
  if (scrollTopBtn) {
    if (window.scrollY > 300) {
      scrollTopBtn.classList.add('visible');
    } else {
      scrollTopBtn.classList.remove('visible');
    }
  }
});

// Intersection Observer for Staggered Scroll Reveal Animations
document.addEventListener('DOMContentLoaded', () => {
  // Automatically add reveal-up class to key section elements if not present
  const targetElements = document.querySelectorAll(
    'section .section-header, .editorial-col, .input-option-card, .workflow-step, .stages-table-card, .doctor-search-panel, .map-card, .contact-form-card, .interactive-pipeline-panel'
  );

  targetElements.forEach((el) => {
    if (!el.classList.contains('reveal-up') && !el.classList.contains('reveal')) {
      el.classList.add('reveal-up');
    }
  });

  // Observe all reveal elements
  const reveals = document.querySelectorAll('.reveal, .reveal-up, .horizontal-workflow');

  const revealObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('active');
        entry.target.classList.add('revealed');

        // Stagger child elements if container has children
        const children = entry.target.querySelectorAll('.editorial-col, .workflow-step, .input-option-card, .trust-pill');
        children.forEach((child, index) => {
          child.style.transitionDelay = `${index * 70}ms`;
          child.classList.add('active');
        });

        observer.unobserve(entry.target);
      }
    });
  }, {
    root: null,
    threshold: 0.15,
    rootMargin: '0px 0px -40px 0px'
  });

  reveals.forEach(el => revealObserver.observe(el));
});

